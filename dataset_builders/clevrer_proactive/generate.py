"""Generate proactive QA from CLEVRER annotations: every question is asked at frame 0, and the answer is a function of the
frames seen so far, A(t) (None = not answerable yet). The shape of A(t) decides the group:
  group 1: None ... None, v, v, ..., v           answer appears at f* and never changes
  group 2: v1 ... v1, v2 ... v2, v3 ...          piecewise constant, an answer per interval
Sources: the official descriptive questions, run on the scene truncated at t (tools/clevrer_exec.py, validated against the
official answers), and new families computed from the per-frame state (visibility, moving, enter / exit / collisions).

Filters: f* (group 1) / first change (group 2) >= MIN_LEAD frames; candidate events closer than TIE frames -> drop; the
series must be the same under neighbouring definitions (moving threshold 0.02 / 0.08 with SHIFT frames tolerance,
collisions with both partners visible); group-2 runs are never merged (that would give real frames a neighbour's
answer), short runs keep their true answer; a group-2 question needs at least one change at >= MIN_LEAD frames.
"Most recent" families answer "none" before their first event.
Choices are always of one kind, so exactly one is correct: 4 colours / 4 counts; 4 objects all written the same way,
'the <color> <shape>' (or 'the <color> <material> <shape>' for all of them when colour + shape is ambiguous in the
scene), never the object named in the question; official shape / material questions keep their wording with the
attribute's values as choices (3 shapes, 2 materials); 2 for yes / no. In questions, objects are referred to by their
shortest description that is unique over the whole video (objects that enter later count).

    python -m dataset_builders.clevrer_proactive.generate --split validation [--scenes 500]   # -> <PROCESSED>/<split>.json
"""
import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import replace

import numpy as np

from .executor import N_FRAMES, Cfg, Executor, Scene, build_cache
from .paths import PROCESSED, RAW

BASE = Cfg(th=0.04, collisions="any")
VARIANTS = [replace(BASE, th=0.02), replace(BASE, th=0.08), replace(BASE, collisions="both")]
MIN_LEAD, TIE, SHIFT, SUSTAIN = 25, 3, 5, 3
COLORS = ["gray", "red", "blue", "green", "brown", "yellow", "cyan", "purple"]
PLURAL = {"cube": "cubes", "sphere": "spheres", "cylinder": "cylinders"}
OUT = PROCESSED


# ---------------------------------------------------------------- scene state
class State:
    """Per-frame state of a scene under one Cfg."""

    def __init__(self, s, cfg):
        self.s, self.cfg = s, cfg
        self.sc = Scene(s, cfg)
        self.n = len(s["objs"])
        self.vis = s["vis"]
        self.mov = self.vis & (self.sc.speed > cfg.th)
        self.events = self.sc.events
        ev = lambda typ: sorted((e["frame"], e["object"]) for e in self.events if e["type"] == typ)
        self.ins, self.outs, self.cols = ev("in"), ev("out"), ev("collision")
        self.starts, self.stops = self._transitions(True), self._transitions(False)

    def _transitions(self, to_moving):
        """(frame, obj) where a visible object switches stationary <-> moving and stays so for SUSTAIN frames."""
        out = []
        for i in range(self.n):
            m, v = self.mov[i], self.vis[i]
            for t in range(SUSTAIN, N_FRAMES - SUSTAIN + 1):
                if v[t - SUSTAIN:t + SUSTAIN].all() and (m[t - SUSTAIN:t] == (not to_moving)).all() and (m[t:t + SUSTAIN] == to_moving).all():
                    out.append((t, [i]))
        return sorted(out)

    def change_points(self):
        cp = {0} | {e["frame"] for e in self.events if e["type"] != "end"}
        for arr in (self.vis, self.mov):
            cp |= set((np.nonzero(np.diff(arr.astype(np.int8), axis=1))[1] + 1).tolist())
        return sorted(t for t in cp if t < N_FRAMES)


class PrefixScene:
    """The scene as seen up to frame h, in the interface clevrer_exec.Executor expects."""

    def __init__(self, st: State, h):
        self.s, self.cfg, self.objs, self.speed = st.s, st.cfg, st.sc.objs, st.sc.speed
        self.h = h
        self.events = [e for e in st.sc.events if e["type"] != "end" and e["frame"] <= h]
        self.traces = [[e for e in tr if e["frame"] <= h] for tr in st.sc.traces]

    def visible(self, o, t=None):
        return bool(self.s["vis"][o, t]) if t is not None else bool(self.s["vis"][o, :self.h + 1].any())

    def moving(self, o, t=None):
        if t is not None:
            return bool(self.speed[o, t] > self.cfg.th)
        return bool((self.s["vis"][o, :self.h + 1] & (self.speed[o, :self.h + 1] > self.cfg.th)).any())


class WorstFutureScene(PrefixScene):
    """Frames up to h as seen, followed by a future where everything that still can happen does: right after h every
    object out of view enters, every object in view exits, every pair collides, and every object moves. An answer that
    is the same here and in the plain prefix (nothing more happens) can no longer change: it is final at h."""

    def __init__(self, st: State, h, phantom=None):
        super().__init__(st, h)
        self.n_real = n = len(self.objs)
        if h < N_FRAMES - 1:
            f, vis = h + 1, st.s["vis"][:, h]
            self.events += [{"type": "out" if vis[o] else "in", "object": [o], "frame": f} for o in range(n)]
            self.events += [{"type": "collision", "object": [a, b], "frame": f} for a in range(n) for b in range(a + 1, n)]
            if phantom:  # an object nobody has seen yet (the viewer does not know the object list) enters and hits everything
                self.s = {**self.s, "objs": self.s["objs"] + [phantom]}
                self.objs = self.objs + [n]
                self.events += [{"type": "in", "object": [n], "frame": f}]
                self.events += [{"type": "collision", "object": [o, n], "frame": f} for o in range(n)]
        self.events.sort(key=lambda e: e["frame"])

    def visible(self, o, t=None):
        if t is None or t > self.h:
            return True
        return False if o >= self.n_real else super().visible(o, t)

    def moving(self, o, t=None):
        if t is None or t > self.h:
            return True
        return False if o >= self.n_real else super().moving(o, t)


def phantoms(program):
    """Unseen objects worth trying: per attribute, the values the program mentions plus one value it does not."""
    vals = {"color": COLORS, "material": ["metal", "rubber"], "shape": ["cube", "sphere", "cylinder"]}
    pick = {}
    for k, vs in vals.items():
        used = [v for v in vs if v in program]
        pick[k] = used + [v for v in vs if v not in used][:1]
    return [{"color": c, "material": m, "shape": s} for c in pick["color"] for m in pick["material"] for s in pick["shape"]]


def is_final(st, t, program, answer):
    """No consistent future changes the answer. Futures that break the program (e.g. a second 'cube' making 'the
    cube' ambiguous) contradict the question's presupposition and are skipped."""
    for ph in [None] + phantoms(program):
        r = Executor(WorstFutureScene(st, t, ph)).run(program)
        if r != "error" and r != answer:
            return False
    return True


def program_series(st: State, program, final=False):
    """A(t) for every frame; 'error' (not answerable yet) -> None. final=True: A(t) only where it can no longer change
    (same result in the plain prefix and in WorstFutureScene), else None -- for yes / no questions without a
    presupposition, e.g. 'Are there any cubes that enter after ...?': 'no' is final only at the end of the video."""
    vals, cps = [], sorted(set(st.change_points()) | ({N_FRAMES - 1} if final else set()))
    for t in cps:
        r = Executor(PrefixScene(st, t)).run(program)
        if final and r != "error" and not is_final(st, t, program, r):
            r = "error"
        vals.append(None if r == "error" else r)
    out, k = [], 0
    for t in range(N_FRAMES):
        while k + 1 < len(cps) and cps[k + 1] <= t:
            k += 1
        out.append(vals[k])
    return out


# ---------------------------------------------------------------- series shapes
def runs(series):
    """[(start, end_inclusive, value)]"""
    out = []
    for t, v in enumerate(series):
        if out and out[-1][2] == v:
            out[-1][1] = t
        else:
            out.append([t, t, v])
    return [tuple(r) for r in out]


def group1(series):
    """f*, value if the series is None* then constant (not None) to the end."""
    rs = runs(series)
    if len(rs) == 1 and rs[0][2] is not None:
        return 0, rs[0][2]
    if len(rs) == 2 and rs[0][2] is None and rs[1][2] is not None:
        return rs[1][0], rs[1][2]
    return None


def group2(series):
    """Intervals (start, end, value) after dropping a leading None. Runs are never merged: that would relabel real
    frames with a neighbour's answer; short runs keep their true answer."""
    rs = runs(series)
    if rs and rs[0][2] is None:
        rs = rs[1:]
    if not rs or any(r[2] is None for r in rs):
        return None
    return rs


def same_g1(a, b, tol):
    return a is not None and b is not None and a[1] == b[1] and abs(a[0] - b[0]) <= tol


def same_g2(a, b, tol):
    return isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and all(x[2] == y[2] and abs(x[0] - y[0]) <= tol for x, y in zip(a, b))


# ---------------------------------------------------------------- language
def describe(objs, i):
    """Shortest description of object i unique among all objects of the video."""
    o = objs[i]
    for keys, noun in ((("shape",), None), (("color",), "object"), (("color", "shape"), None), (("material", "shape"), None),
                       (("color", "material"), "object"), (("color", "material", "shape"), None)):
        if sum(all(p[k] == o[k] for k in keys) for p in objs) == 1:
            words = [o[k] for k in keys if k != "shape"] + [o["shape"] if "shape" in keys else noun]
            return " ".join(words)
    return f"{o['color']} {o['material']} {o['shape']}"


def rng_for(key):
    return random.Random(int(hashlib.md5(key.encode()).hexdigest()[:8], 16))


def color_choices(objs, answers, key, exclude=(), n=4):
    """answers (1..n colours) + colours of other objects in the scene, then random colours; n colours in total."""
    rng = rng_for(key)
    ch = list(dict.fromkeys(answers))
    scene = [o["color"] for k, o in enumerate(objs) if k not in exclude and o["color"] not in ch]
    rng.shuffle(scene)
    for c in scene + rng.sample(COLORS, len(COLORS)):
        if len(ch) >= n:
            break
        if c not in ch:
            ch.append(c)
    rng.shuffle(ch)
    return ch


def object_label(o, full):
    return f"the {o['color']} {o['material']} {o['shape']}" if full else f"the {o['color']} {o['shape']}"


def object_choices(objs, answer, key, exclude=()):
    """(choices, answer label): every choice has the same form, 'the <color> <shape>', or 'the <color> <material>
    <shape>' for all of them when colour + shape does not identify every object of the scene. Distractors: other
    objects of the scene (not the answer, not objects named in the question), then combinations absent from the scene."""
    rng = rng_for(key)
    full = len({(o["color"], o["shape"]) for o in objs}) < len(objs)
    keyf = (lambda o: (o["color"], o["material"], o["shape"])) if full else (lambda o: (o["color"], o["shape"]))
    ans = object_label(objs[answer], full)
    others = [object_label(objs[k], full) for k in range(len(objs)) if k != answer and k not in exclude]
    rng.shuffle(others)
    scene = {keyf(o) for o in objs}
    fakes = [{"color": c, "material": m, "shape": s} for c in COLORS for m in ("metal", "rubber") for s in PLURAL]
    fakes = list(dict.fromkeys(object_label(f, full) for f in fakes if keyf(f) not in scene))
    rng.shuffle(fakes)
    ch = [ans] + others[:3]
    ch += fakes[:4 - len(ch)]
    rng.shuffle(ch)
    return ch, ans


def count_choices(values, key):
    vals = sorted(set(int(v) for v in values))
    if len(vals) > 4:
        return None
    lo, hi = min(vals), max(vals)
    cand = sorted(set(range(0, 10)) - set(vals), key=lambda x: (min(abs(x - lo), abs(x - hi)), x))
    ch = vals + cand[:4 - len(vals)]
    rng_for(key).shuffle(ch)
    return [str(c) for c in ch]


# ---------------------------------------------------------------- families
def first_event(events, k=0):
    """series: None until the k-th event, then its object; None if the k-th event is tied with a neighbour."""
    if len(events) <= k:
        return None
    f = events[k][0]
    if any(abs(e[0] - f) < TIE for j, e in enumerate(events) if j != k):
        return None
    return [None] * f + [events[k][1][0]] * (N_FRAMES - f)


def partner_events(st, x):
    return [(f, [b if a == x else a]) for f, (a, b) in st.cols if x in (a, b)]


G1_OBJ = {  # name -> (series fn(state, x), color template, object template, needs x)
    "first_enter": (lambda st, x: first_event(st.ins), "What color is the first object that enters the scene?", "Which object enters the scene first?", False),
    "second_enter": (lambda st, x: first_event(st.ins, 1), "What color is the second object that enters the scene?", "Which object is the second to enter the scene?", False),
    "first_exit": (lambda st, x: first_event(st.outs), "What color is the first object that leaves the scene?", "Which object leaves the scene first?", False),
    # only objects at rest (visible, not moving) at the start: an object already rolling at frame 0 would otherwise
    # be the natural reading of "starts moving first"
    "first_start": (lambda st, x: first_event([e for e in st.starts if st.vis[e[1][0], 0] and not st.mov[e[1][0], :SUSTAIN].any()]),
                    "What color is the first of the still objects to start moving?",
                    "Which of the still objects starts moving first?", False),
    "first_stop": (lambda st, x: first_event(st.stops), "What color is the first object that comes to a stop?", "Which object comes to a stop first?", False),
    "first_hit": (lambda st, x: first_event(partner_events(st, x)), "What color is the first object that collides with the {x}?", "Which object is the first to collide with the {x}?", True),
}


def yesno_series(frames):
    """yes from the first event frame on; no is known only at the end of the video."""
    if frames:
        return [None] * frames[0] + ["yes"] * (N_FRAMES - frames[0])
    return [None] * (N_FRAMES - 1) + ["no"]


G1_YESNO = {  # name -> (fn(state, x, y) -> event frames, template, applies(st0, x, y))
    "will_collide": (lambda st, x, y: [f for f, (a, b) in st.cols if {a, b} == {x, y}], "Will the {x} collide with the {y}?", lambda st, x, y: x < y),
    "will_exit": (lambda st, x, y: [f for f, (o,) in st.outs if o == x], "Will the {x} leave the scene?", lambda st, x, y: y is None and st.vis[x, 0]),
    "will_start": (lambda st, x, y: [f for f, (o,) in st.starts if o == x], "Will the {x} start moving?",
                   lambda st, x, y: y is None and st.vis[x, 0] and not st.mov[x, :SUSTAIN].any()),
}

G2 = {  # name -> (fn(state, x) -> series, template, answer type)
    "n_visible": (lambda st, x: [str(int(v)) for v in st.vis.sum(0)], "How many objects are in view right now?", "count"),
    "n_moving": (lambda st, x: [str(int(v)) for v in st.mov.sum(0)], "How many objects are moving right now?", "count"),
    "n_stationary": (lambda st, x: [str(int(v)) for v in (st.vis & ~st.mov).sum(0)], "How many objects are standing still in view right now?", "count"),
    "n_collisions": (lambda st, x: [str(sum(f <= t for f, _ in st.cols)) for t in range(N_FRAMES)], "How many collisions have happened so far?", "count"),
    # before the first event the answer is "none" (no object has entered / hit X yet), not "unknown"
    "last_enter": (lambda st, x: [next((o[0] for f, o in reversed(st.ins) if f <= t), "none") for t in range(N_FRAMES)],
                   "What color is the object that entered the scene most recently?", "color_obj"),
    "last_hit": (lambda st, x: [next((o[0] for f, o in reversed(partner_events(st, x)) if f <= t), "none") for t in range(N_FRAMES)],
                 "What color is the last object that has collided with the {x} so far?", "color_obj"),
    "is_moving": (lambda st, x: ["yes" if m else "no" for m in st.mov[x]], "Is the {x} moving right now?", "yesno"),
}


def n_visible_attr(st, key, val):
    sel = np.array([o[key] == val for o in st.s["objs"]])
    return [str(int(v)) for v in st.vis[sel].sum(0)]


# ---------------------------------------------------------------- generation
def ev_key(e):
    return e["type"], tuple(sorted(e.get("object", ())))


def causes(st: State, key):
    """(frame, {ancestor event keys}) of the event `key` in the causal graph of the official executor
    (filter_ancestor: earlier events along the objects' event traces), or None if the event does not happen."""
    ev = next((e for e in st.sc.events if ev_key(e) == key), None)
    if ev is None:
        return None
    return ev["frame"], {ev_key(e) for e in Executor(st.sc).m_filter_ancestor(st.sc.events, ev)}


def generate_scene(sid, s, split, qs, drops):
    states = [State(s, c) for c in [BASE] + VARIANTS]
    st, objs, n = states[0], s["objs"], len(s["objs"])
    desc = [describe(objs, i) for i in range(n)]
    vid = f"video_{sid:05d}.mp4"
    out = []

    def emit(group, family, source, question, atype, choices, answers, extra=None):
        out.append({"id": f"{split}_{sid}_{family}_{len(out)}", "video": vid, "scene_index": sid, "split": split, "group": group,
                    "family": family, "source": source, "question": question, "answer_type": atype, "choices": choices,
                    "ask_frame": 0, "answers": answers, **(extra or {})})

    def robust_g1(fn):
        a = group1(fn(states[0])) if fn(states[0]) is not None else None
        if a is None:
            return None, "not_g1"
        for k, v in enumerate(states[1:]):
            sv = fn(v)
            if sv is None or not same_g1(a, group1(sv), SHIFT if k < 2 else 0):
                return None, "not_robust"
        return a, None

    def robust_g2(fn):
        a = group2(fn(states[0]))
        if a is None:
            return None, "not_g2"
        for k, v in enumerate(states[1:]):
            if not same_g2(a, group2(fn(v)), SHIFT if k < 2 else 0):
                return None, "not_robust"
        return a, None

    def g1_obj(family, source, fn, q_color, q_obj, exclude=()):
        """Object-valued group-1 series, asked either as its colour or as 'which object' (uniform object choices)."""
        a, why = robust_g1(fn)
        if why:
            drops[(family, why)] += 1
            return
        f, o = a
        if f < MIN_LEAD:
            drops[(family, "too_early")] += 1
            return
        o = int(o)
        key = f"{sid}|{family}|{q_color}"
        if rng_for(key).random() < 0.5:
            emit(1, family, source, q_color, "color", color_choices(objs, [objs[o]["color"]], key, exclude),
                 [{"start_frame": f, "end_frame": N_FRAMES - 1, "answer": objs[o]["color"]}])
        else:
            ch, ans = object_choices(objs, o, key, exclude)
            emit(1, family, source, q_obj, "object", ch, [{"start_frame": f, "end_frame": N_FRAMES - 1, "answer": ans}])

    # --- official descriptive questions, run on the truncated scene; asked as written, choices = the attribute's values
    ATTR_CHOICES = {"query_shape": ["cube", "sphere", "cylinder"], "query_material": ["metal", "rubber"]}
    for q in qs:
        if q["question_type"] != "descriptive" or "end" in q["program"]:
            continue
        prog, sub, fam = q["program"], q["question_subtype"], "clevrer_" + q["question_subtype"]
        src = f"clevrer:{q['question_id']}"
        if sub == "count":  # never "not answerable yet", a running count is not the asked total
            drops[(fam, "count_total")] += 1
            continue
        # query_*: the question presupposes its object (the object to collide with ...), so the answer counts once it
        # is identified; exist has no presupposition: its answer counts only once no future can change it
        a, why = robust_g1(lambda v, p=prog: program_series(v, p, final=sub == "exist"))
        if why or a[0] < MIN_LEAD:
            drops[(fam, why or "too_early")] += 1
            continue
        f, ans = a
        key = f"{sid}|{src}"
        extra = None
        if sub == "query_color":
            atype, ch = "color", color_choices(objs, [ans], key)
        elif sub in ATTR_CHOICES:
            atype, ch = sub[6:], ATTR_CHOICES[sub][:]
            rng_for(key).shuffle(ch)
        else:  # exist
            atype, ch, extra = "yesno", ["yes", "no"], {"resolved_at_end": f == N_FRAMES - 1}
        emit(1, fam, src, q["question"], atype, ch, [{"start_frame": f, "end_frame": N_FRAMES - 1, "answer": ans}], extra)

    # --- new group-1 families: first X
    for fam, (fn, qc, qo, needs_x) in G1_OBJ.items():
        for x in (range(n) if needs_x else [None]):
            g1_obj(fam, "new", lambda v, f=fn, x=x: f(v, x), qc.format(x=desc[x] if x is not None else ""),
                   qo.format(x=desc[x] if x is not None else ""), exclude=(x,) if x is not None else ())

    # --- new group-1 yes / no: will X ...; "no" pairs are subsampled to the number of "yes" ones
    for fam, (fn, tmpl, ok) in G1_YESNO.items():
        cands = [(x, y) for x in range(n) for y in ([None] if fam != "will_collide" else range(n)) if ok(st, x, y)]
        res = {"yes": [], "no": []}
        for x, y in cands:
            a, why = robust_g1(lambda v, x=x, y=y: yesno_series(fn(v, x, y)))
            if why:
                drops[(fam, why)] += 1
                continue
            if a[1] == "yes" and a[0] < MIN_LEAD:
                drops[(fam, "too_early")] += 1
                continue
            res[a[1]].append((x, y, a))
        rng = rng_for(f"{sid}|{fam}")
        rng.shuffle(res["no"])
        for x, y, (f, ans) in res["yes"] + res["no"][:max(1, len(res["yes"]))]:
            emit(1, fam, "new", tmpl.format(x=desc[x], y=desc[y] if y is not None else ""), "yesno", ["yes", "no"],
                 [{"start_frame": f, "end_frame": N_FRAMES - 1, "answer": ans}], {"resolved_at_end": ans == "no"})

    # --- group 1, explanatory: what is responsible for a collision / an exit. f* = the event's frame (its causes are
    # all earlier). One correct choice + 3 wrong ones, all of one kind: presences of objects, collisions or entrances.
    col_keys = {("collision", tuple(sorted(o))) for v in states for _, o in v.cols}
    in_keys = {("in", tuple(o)) for _, o in st.ins}
    targets = [(("collision", tuple(sorted(o))), "cause_collision") for _, o in st.cols] + [(("out", tuple(o)), "cause_exit") for _, o in st.outs]
    for key, fam in targets:
        res = [causes(v, key) for v in states]
        if res[0] is None or not res[0][1]:
            drops[(fam, "no_cause")] += 1
            continue
        if any(r is None or r[1] != res[0][1] or abs(r[0] - res[0][0]) > SHIFT for r in res[1:]):
            drops[(fam, "not_robust")] += 1
            continue
        f, anc = res[0]
        if f < MIN_LEAD:
            drops[(fam, "too_early")] += 1
            continue
        tobj = set(key[1])
        anc_objs = {o for _, os_ in anc for o in os_} - tobj
        name = {
            "presence": lambda o: f"the presence of the {desc[o]}",
            "collision": lambda k: f"the collision between the {desc[k[1][0]]} and the {desc[k[1][1]]}",
            "entering": lambda k: f"the {desc[k[1][0]]}'s entering the scene",
        }
        never_in = [("in", (o,)) for o in range(n) if st.vis[o, 0] and ("in", (o,)) not in in_keys]
        pairs = [("collision", (a, b)) for a in range(n) for b in range(a + 1, n) if ("collision", (a, b)) not in col_keys]
        kinds = {  # kind -> (correct options, wrong real ones, wrong ones that never happen)
            "presence": (sorted(anc_objs), [o for o in range(n) if o not in anc_objs and o not in tobj], []),
            "collision": (sorted(k for k in anc if k[0] == "collision"), sorted(col_keys - anc - {key}), pairs),
            "entering": (sorted(k for k in anc if k[0] == "in"), sorted(in_keys - anc), never_in),
        }
        rng = rng_for(f"{sid}|{fam}|{key}")
        # at most one wrong option that never happens (it is ruled out just by looking); the others are real events
        feasible = [k for k, (c, w, fk) in kinds.items() if c and len(w) + min(1, len(fk)) >= 3]
        if not feasible:
            drops[(fam, "few_options")] += 1
            continue
        kind = rng.choice(feasible)
        corr, wrong, fake = kinds[kind]
        wrong, fake = wrong[:], fake[:]
        rng.shuffle(wrong)
        rng.shuffle(fake)
        ans = name[kind](rng.choice(corr))
        ch = [ans] + [name[kind](w) for w in (wrong + fake[:1])[:3]]
        rng.shuffle(ch)
        if key[0] == "collision":
            q = f"Which of the following is responsible for the collision between the {desc[key[1][0]]} and the {desc[key[1][1]]}?"
        else:
            q = f"Which of the following is responsible for the {desc[key[1][0]]}'s exit?"
        emit(1, fam, "new", q, "event", ch, [{"start_frame": f, "end_frame": N_FRAMES - 1, "answer": ans}], {"cause_kind": kind})

    # --- group 2
    g2_items = [(fam, fn, tmpl, at, None) for fam, (fn, tmpl, at) in G2.items() if "{x}" not in tmpl]
    g2_items += [(fam, fn, tmpl, at, x) for fam, (fn, tmpl, at) in G2.items() if "{x}" in tmpl for x in range(n)]
    for key in ("shape", "color", "material"):
        for val in sorted({o[key] for o in objs}):
            if sum(o[key] == val for o in objs) >= 2:
                noun = PLURAL[val] if key == "shape" else f"{val} objects"
                g2_items.append(("n_visible_attr", lambda v, x, k=key, vl=val: n_visible_attr(v, k, vl), f"How many {noun} are in view right now?", "count", None))
    for fam, fn, tmpl, at, x in g2_items:
        if fam == "is_moving" and not st.vis[x].all():
            continue
        iv, why = robust_g2(lambda v, f=fn, x=x: f(v, x))
        if why:
            drops[(fam, why)] += 1
            continue
        if len(iv) < 2:
            drops[(fam, "one_interval")] += 1
            continue
        # proactive: at least one change of the answer at >= MIN_LEAD frames
        if not any(r[0] >= MIN_LEAD for r in iv[1:]):
            drops[(fam, "too_early")] += 1
            continue
        key = f"{sid}|{fam}|{tmpl}|{x}"
        q = tmpl.format(x=desc[x] if x is not None else "")
        if at == "count":
            ch = count_choices([v for _, _, v in iv], key)
            ans = [v for _, _, v in iv]
        elif at == "yesno":
            ch, ans = ["yes", "no"], [v for _, _, v in iv]
        else:  # colour of an object, or "none" before the first event
            ans = [v if v == "none" else objs[int(v)]["color"] for _, _, v in iv]
            if sum(a != "none" for a in ans) < 2:  # "none -> X" alone is the group-1 first_hit / first_enter question
                drops[(fam, "dup_of_group1")] += 1
                continue
            if any(a == b for a, b in zip(ans, ans[1:])):
                drops[(fam, "same_colour_twice")] += 1
                continue
            colors = [a for a in ans if a != "none"]
            ch = None
            if len(set(ans)) <= 4:
                ch = color_choices(objs, colors, key, exclude=(x,) if x is not None else (), n=4 - ("none" in ans))
                if "none" in ans:
                    ch.insert(rng_for(key + "none").randrange(4), "none")
        if ch is None:
            drops[(fam, "too_many_values")] += 1
            continue
        emit(2, fam, "new", q, "yesno" if at == "yesno" else ("count" if at == "count" else "color"), ch,
             [{"start_frame": a, "end_frame": b, "answer": v} for (a, b, _), v in zip(iv, ans)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--scenes", type=int, default=None)
    args = ap.parse_args()

    cache = build_cache(args.split)
    qfile = json.loads((RAW / f"{args.split}.json").read_text())[:args.scenes]
    drops, out = Counter(), []
    for k, v in enumerate(qfile):
        out += generate_scene(v["scene_index"], cache[v["scene_index"]], args.split, v["questions"], drops)
        if (k + 1) % 100 == 0:
            print(f"{k + 1} scenes, {len(out)} questions", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"{args.split}{'_' + str(args.scenes) if args.scenes else ''}.json"
    (OUT / name).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    n = len(qfile)
    print(f"\n{len(out)} questions on {n} scenes ({len(out) / n:.1f} per scene) -> {OUT / name}")
    fam = Counter((q["group"], q["family"]) for q in out)
    print("\nper family (kept / dropped by reason):")
    for (g, f), c in sorted(fam.items()):
        dr = {r: d for (ff, r), d in drops.items() if ff == f}
        print(f"  g{g} {f:22} {c:6}  ({c / n:.2f}/scene)  drops {dict(dr)}")
    for (f, r), d in sorted(drops.items()):
        if (1, f) not in fam and (2, f) not in fam:
            print(f"  -- {f:22} 0  drops {r}={d}")
    g1 = [q for q in out if q["group"] == 1]
    fs = [q["answers"][0]["start_frame"] for q in g1 if not q.get("resolved_at_end")]
    print("\ngroup 1 f* (s):", np.percentile(np.array(fs) / 25, [10, 25, 50, 75, 90]).round(2), " resolved at end (no):", sum(q.get("resolved_at_end", False) for q in g1))
    g2 = [q for q in out if q["group"] == 2]
    print("group 2 intervals per question:", sorted(Counter(len(q["answers"]) for q in g2).items()))
    print("answer types:", Counter(q["answer_type"] for q in out))
    print("yes/no balance:", Counter(a["answer"] for q in out if q["answer_type"] == "yesno" for a in q["answers"]))


if __name__ == "__main__":
    main()
