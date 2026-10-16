"""Symbolic executor for CLEVRER question programs on the ground-truth annotations, with the ambiguous definitions
exposed as config knobs, so they can be checked against the official answers (tools/clevrer_validate.py).

Programs in questions/<split>.json are postfix lists in the generator vocabulary (get_frame, get_col_partner,
get_object, start / end / null ...). Semantics follow the official NS-DR executor (chuangg/CLEVRER executor/*.py):
  - moving with a frame: speed at that frame > th;  without a frame (null): moving at some frame while visible
  - stationary with a frame: visible and not moving;  without: visible at some frame and never moving while visible
  - in / out: visibility transitions between sampled frames (an object visible from the first frame never 'enters')
  - events: start (frame 0), end (Cfg.end_frame), in / out, collisions; before / after are strict
  - explanatory: causal ancestors along per-object event traces; a choice is correct if
    run(choice_program + question_program) == 'yes'
Predictive / counterfactual programs need unseen / counterfactual simulations that are not released and are skipped.

Annotations are cached once per split as compact arrays: <CACHE>/cache_<split>.pkl (see paths.py).
"""
import glob
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paths import CACHE, RAW

COLORS = {"gray", "red", "blue", "green", "brown", "yellow", "cyan", "purple"}
MATERIALS = {"metal", "rubber"}
SHAPES = {"sphere", "cylinder", "cube"}
N_FRAMES = 128


@dataclass(frozen=True)
class Cfg:
    speed: str = "vel"          # 'vel': |annotated velocity xy| (units / s); 'disp': |location change| per frame * 35.6
    th: float = 0.1             # moving threshold, units / s
    step: int = 1               # frame sampling for in / out detection (official code: 5)
    speed_step: int = 5         # 'disp' speed: displacement over this many frames (official code: 5)
    end_frame: int = 125        # frame of the 'end' event (official code: 125)
    collisions: str = "all"     # 'all' | 'any' visible partner | 'both' visible
    ties: str = "stable"        # order of events on the same frame: 'stable' (as listed) | 'objid'


def build_cache(split):
    """scene_index -> dict(objs, vis[n,128] bool, vel[n,128,2], loc[n,128,2], cols[(a, b, frame)])"""
    path = CACHE / f"cache_{split}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    cache = {}
    for p in sorted(glob.glob(str(RAW / "annotations" / split / "*" / "annotation_*.json"))):
        sid, s = load_annotation(p)
        cache[sid] = s
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(cache, protocol=pickle.HIGHEST_PROTOCOL))
    return cache


def annotation_path(split, sid):
    """<RAW>/annotations/<split>/annotation_10000-11000/annotation_10001.json"""
    lo = sid // 1000 * 1000
    return RAW / "annotations" / split / f"annotation_{lo:05d}-{lo + 1000:05d}" / f"annotation_{sid:05d}.json"


def load_annotation(p):
    """One annotation file -> (scene_index, cache entry as in build_cache)."""
    a = json.loads(Path(p).read_text())
    n = len(a["object_property"])
    vis = np.zeros((n, N_FRAMES), bool)
    vel = np.zeros((n, N_FRAMES, 2), np.float32)
    loc = np.zeros((n, N_FRAMES, 2), np.float32)
    for f in a["motion_trajectory"]:
        for o in f["objects"]:
            i, t = o["object_id"], f["frame_id"]
            vis[i, t] = o["inside_camera_view"]
            vel[i, t] = o["velocity"][:2]
            loc[i, t] = o["location"][:2]
    return a["scene_index"], {"objs": [{k: o[k] for k in ("color", "material", "shape")} for o in a["object_property"]],
                              "vis": vis, "vel": vel, "loc": loc,
                              "cols": [(c["object_ids"][0], c["object_ids"][1], c["frame_id"]) for c in a["collision"]]}


class Scene:
    def __init__(self, s, cfg: Cfg):
        self.s, self.cfg = s, cfg
        self.objs = list(range(len(s["objs"])))
        self.frames = list(range(0, N_FRAMES, cfg.step))
        self.speed = self._speed()
        self.events = self._events()
        self.traces = [sorted((e for e in self.events if e["type"] not in ("start", "end") and o in e["object"]), key=lambda e: e["frame"])
                       for o in self.objs]

    def _speed(self):
        if self.cfg.speed == "vel":
            return np.linalg.norm(self.s["vel"], axis=2)
        # official style: average of backward / forward displacement over `speed_step`, only to visible neighbours
        loc, vis, k = self.s["loc"], self.s["vis"], self.cfg.speed_step
        sp = np.zeros(vis.shape, np.float32)
        for i in self.objs:
            for t in range(N_FRAMES):
                ds = []
                if t - k >= 0 and vis[i, t - k]:
                    ds.append((loc[i, t] - loc[i, t - k]) / k)
                if t + k < N_FRAMES and vis[i, t + k]:
                    ds.append((loc[i, t + k] - loc[i, t]) / k)
                sp[i, t] = np.linalg.norm(np.mean(ds, axis=0)) * 35.6 if ds else 0.0
        return sp

    def _events(self):
        vis, ev = self.s["vis"], [{"type": "start", "frame": 0}, {"type": "end", "frame": self.cfg.end_frame}]
        fr = self.frames
        for i in self.objs:
            for j, t in enumerate(fr):
                if not vis[i, t]:
                    continue
                if j > 0 and not vis[i, fr[j - 1]]:
                    ev.append({"type": "in", "object": [i], "frame": t})
                if j < len(fr) - 1 and not vis[i, fr[j + 1]]:
                    ev.append({"type": "out", "object": [i], "frame": fr[j + 1]})
        for a, b, t in self.s["cols"]:
            if t >= N_FRAMES:
                continue
            va, vb = vis[a, min(t, N_FRAMES - 1)], vis[b, min(t, N_FRAMES - 1)]
            if self.cfg.collisions == "both" and not (va and vb):
                continue
            if self.cfg.collisions == "any" and not (va or vb):
                continue
            ev.append({"type": "collision", "object": [a, b], "frame": t})
        key = (lambda e: e["frame"]) if self.cfg.ties == "stable" else (lambda e: (e["frame"], e.get("object", [-1])))
        return sorted(ev, key=key)

    # --- state ---
    def visible(self, o, t=None):
        return bool(self.s["vis"][o, t]) if t is not None else bool(self.s["vis"][o].any())

    def moving(self, o, t=None):
        if t is not None:
            return bool(self.speed[o, min(t, N_FRAMES - 1)] > self.cfg.th)
        return any(self.s["vis"][o, f] and self.speed[o, f] > self.cfg.th for f in self.frames)


class Executor:
    NARGS = {"objects": 0, "events": 0, "start": 0, "end": 0, "unique": 1, "count": 1, "exist": 1, "negate": 1,
             "belong_to": 2, "filter_color": 2, "filter_material": 2, "filter_shape": 2, "filter_moving": 2,
             "filter_stationary": 2, "filter_in": 2, "filter_out": 2, "filter_collision": 2, "filter_order": 2,
             "filter_before": 2, "filter_after": 2, "get_frame": 1, "get_object": 1, "get_col_partner": 2,
             "query_color": 1, "query_material": 1, "query_shape": 1, "filter_ancestor": 2}

    def __init__(self, scene: Scene):
        self.sc = scene

    def run(self, program):
        stack = []
        for m in program:
            if m not in self.NARGS:
                stack.append(None if m == "null" else m)  # literal: attribute value, first / second / last, null
                continue
            n = self.NARGS[m]
            if len(stack) < n:
                return "error"
            args = stack[len(stack) - n:] if n else []
            del stack[len(stack) - n:]
            try:
                out = getattr(self, "m_" + m)(*args)
            except (TypeError, IndexError, KeyError, ValueError):
                return "error"
            if out == "error":
                return "error"
            stack.append(out)
        return str(stack[-1]) if stack else "error"

    # sets
    def m_objects(self): return self.sc.objs[:]
    def m_events(self): return self.sc.events[:]
    def m_start(self): return self.sc.events[0] if self.sc.events[0]["type"] == "start" else next(e for e in self.sc.events if e["type"] == "start")
    def m_end(self): return next(e for e in self.sc.events if e["type"] == "end")

    def m_unique(self, xs): return xs[0] if isinstance(xs, list) and len(xs) == 1 else "error"
    def m_count(self, xs): return len(xs) if isinstance(xs, list) else "error"
    def m_exist(self, xs): return ("yes" if xs else "no") if isinstance(xs, list) else "error"
    def m_negate(self, b): return {"yes": "no", "no": "yes"}.get(b, "error")

    def m_belong_to(self, entry, events):
        evs = [e for e in events if e["type"] not in ("start", "end")]
        if isinstance(entry, dict):
            return "yes" if any(e["type"] == entry["type"] and set(e["object"]) == set(entry["object"]) for e in evs) else "no"
        if isinstance(entry, int):
            return "yes" if any(entry in e["object"] for e in evs) else "no"
        return "error"

    # object filters
    def _attr(self, objs, key, val, allowed):
        if val not in allowed or not isinstance(objs, list):
            return "error"
        return [o for o in objs if self.sc.s["objs"][o][key] == val]

    def m_filter_color(self, objs, v): return self._attr(objs, "color", v, COLORS)
    def m_filter_material(self, objs, v): return self._attr(objs, "material", v, MATERIALS)
    def m_filter_shape(self, objs, v): return self._attr(objs, "shape", v, SHAPES)

    def m_filter_moving(self, objs, t):
        return [o for o in objs if self.sc.visible(o, t) and self.sc.moving(o, t)]

    def m_filter_stationary(self, objs, t):
        return [o for o in objs if self.sc.visible(o, t) and not self.sc.moving(o, t)]

    # event filters
    def _ev(self, events, typ, objs):
        objs = objs if isinstance(objs, list) else [objs]
        return [e for e in events if e["type"] == typ and any(o in e["object"] for o in objs)]

    def m_filter_in(self, events, objs): return self._ev(events, "in", objs)
    def m_filter_out(self, events, objs): return self._ev(events, "out", objs)
    def m_filter_collision(self, events, objs): return self._ev(events, "collision", objs)

    def m_filter_order(self, events, order):
        idx = {"first": 0, "second": 1, "last": -1}[order]
        return events[idx] if events and (idx < len(events)) else "error"

    def m_filter_before(self, events, ev): return [e for e in events if e["frame"] < ev["frame"]]
    def m_filter_after(self, events, ev): return [e for e in events if e["frame"] > ev["frame"]]

    def m_get_frame(self, ev): return ev["frame"]
    def m_get_object(self, ev): return ev["object"][0] if ev["type"] in ("in", "out") else "error"

    def m_get_col_partner(self, ev, o):
        if ev["type"] != "collision" or o not in ev["object"]:
            return "error"
        a, b = ev["object"]
        return b if o == a else a

    def m_query_color(self, o): return self.sc.s["objs"][o]["color"]
    def m_query_material(self, o): return self.sc.s["objs"][o]["material"]
    def m_query_shape(self, o): return self.sc.s["objs"][o]["shape"]

    # explanatory
    def m_filter_ancestor(self, events, ev):
        causes = []

        def search(target):
            nxt = []
            for tr in self.sc.traces:
                if target in tr:
                    k = tr.index(target)
                    if k > 0 and tr[k - 1] not in nxt:
                        nxt.append(tr[k - 1])
                        if tr[k - 1] not in causes:
                            causes.append(tr[k - 1])
            for e in nxt:
                search(e)

        search(ev)
        return [e for e in events if e in causes]
