"""Check executor.py definitions against the official CLEVRER answers.

Runs the programs of the descriptive questions (and explanatory choices: choice program + question program -> yes /
no vs correct / wrong) with a given Cfg and reports accuracy overall, per question kind and per group of questions
whose program depends on one knob only:
  moving   filter_moving / filter_stationary, no in / out / collision / order    -> speed source, th
  end      programs with 'end' or 'start' (state at the first / last frame)      -> end_frame, th
  inout    filter_in / filter_out, no moving filters                             -> step
  coll     filter_collision, no in / out / order / moving                        -> collisions
  order    filter_order / filter_before / filter_after                           -> ties, step

    python -m dataset_builders.clevrer_proactive.validate sweep  --split train [--scenes 2000]
    python -m dataset_builders.clevrer_proactive.validate errors --split validation [--n 40] [cfg overrides, e.g. th=0.2 step=5]
"""
import argparse
import json
from collections import Counter, defaultdict
from dataclasses import replace

from .executor import Cfg, Executor, Scene, build_cache
from .paths import RAW

GROUPS = {
    "moving": lambda p: ({"filter_moving", "filter_stationary"} & p) and not ({"filter_in", "filter_out", "filter_collision", "filter_order", "filter_before", "filter_after"} & p),
    "end": lambda p: bool({"end", "start"} & p),
    "inout": lambda p: ({"filter_in", "filter_out"} & p) and not ({"filter_moving", "filter_stationary"} & p),
    "coll": lambda p: ("filter_collision" in p) and not ({"filter_in", "filter_out", "filter_order", "filter_before", "filter_after", "filter_moving", "filter_stationary"} & p),
    "order": lambda p: bool({"filter_order", "filter_before", "filter_after"} & p),
}


def items(split, scenes=None):
    """(scene_index, kind, program, expected, question text)"""
    qs = json.loads((RAW / f"{split}.json").read_text())
    out = []
    for v in qs[:scenes] if scenes else qs:
        for q in v["questions"]:
            if q["question_type"] == "descriptive":
                out.append((v["scene_index"], q.get("question_subtype"), q["program"], str(q["answer"]), q["question"]))
            elif q["question_type"] == "explanatory":
                for c in q["choices"]:
                    out.append((v["scene_index"], "explanatory", c["program"] + q["program"],
                                "yes" if c["answer"] == "correct" else "no", q["question"] + " || " + c["choice"]))
    return out


def evaluate(cache, its, cfg, want_errors=0):
    stats = defaultdict(Counter)
    errors, scenes = [], {}
    for sid, kind, prog, exp, text in its:
        if sid not in scenes:
            scenes = {sid: Executor(Scene(cache[sid], cfg))}  # items are grouped by scene
        got = scenes[sid].run(prog)
        ok = got == exp
        p = set(prog)
        keys = ["all", kind] + [g for g, f in GROUPS.items() if f(p)]
        for k in keys:
            stats[k]["n"] += 1
            stats[k]["ok"] += ok
            stats[k]["err"] += got == "error"
        if not ok and len(errors) < want_errors:
            errors.append({"scene": sid, "kind": kind, "q": text, "expected": exp, "got": got, "program": " ".join(prog)})
    return stats, errors


def fmt(stats, keys=("all", "moving", "end", "inout", "coll", "order", "count", "exist", "query_color", "query_shape", "query_material", "explanatory")):
    return "  ".join(f"{k} {100 * stats[k]['ok'] / stats[k]['n']:.2f}" for k in keys if stats[k]["n"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sweep", "errors", "eval"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--scenes", type=int, default=None)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("overrides", nargs="*", help="Cfg fields, e.g. th=0.2 step=5 speed=disp")
    args = ap.parse_args()

    cache = build_cache(args.split)
    its = items(args.split, args.scenes)
    base = Cfg()
    for kv in args.overrides:
        k, v = kv.split("=")
        base = replace(base, **{k: type(getattr(base, k))(v)})
    print(f"{args.split}: {len(its)} items on {len({i[0] for i in its})} scenes; group sizes:",
          {g: sum(bool(f(set(i[2]))) for i in its) for g, f in GROUPS.items()}, flush=True)

    if args.mode == "sweep":
        grid = [("th", [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]), ("speed", ["vel", "disp"]), ("step", [1, 5]),
                ("end_frame", [125, 127]), ("collisions", ["all", "any", "both"]), ("ties", ["stable", "objid"])]
        for knob, vals in grid:
            for val in vals:
                cfg = replace(base, **{knob: val})
                stats, _ = evaluate(cache, its, cfg)
                print(f"{knob}={val!s:7} | {fmt(stats)}", flush=True)
    else:
        stats, errors = evaluate(cache, its, base, args.n if args.mode == "errors" else 0)
        print(base)
        print(fmt(stats))
        print("errors (executor returned 'error'):", {k: c["err"] for k, c in stats.items() if c["err"]})
        for e in errors:
            print(json.dumps(e, ensure_ascii=False))


if __name__ == "__main__":
    main()
