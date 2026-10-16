"""A first training subset of <split>_balanced.json: group 1 only, no cause-effect questions, no "no" that is known only
at the end of the video. Written in both formats: <PROCESSED>/<name>.json (generator format, frames) and
<PROCESSED>/<name>.jsonl (common schema, seconds; as export.py).

Dropped:
  - group 2 (an answer per interval);
  - cause_collision, cause_exit;
  - will_collide, will_exit, will_start entirely: their "no" is known only at the end, and without it every answer is "yes";
  - clevrer_exist answered only at the end (resolved_at_end).
In what is left an answer appears at an event inside the video and does not change.

    python -m dataset_builders.clevrer_proactive.subset            # -> train_group1.*, val_group1.*
"""
import json

from online_vllm.data.schema import save_jsonl

from .balance import stats
from .export import convert
from .paths import PROCESSED

DROP_FAMILIES = {"cause_collision", "cause_exit", "will_collide", "will_exit", "will_start"}
SPLITS = {"train": "train_group1", "validation": "val_group1"}


def keep(q):
    return q["group"] == 1 and q["family"] not in DROP_FAMILIES and not q.get("resolved_at_end")


def main():
    for split, name in SPLITS.items():
        qs = json.loads((PROCESSED / f"{split}_balanced.json").read_text(encoding="utf-8"))
        kept = [q for q in qs if keep(q)]
        stats(kept, name)
        (PROCESSED / f"{name}.json").write_text(json.dumps(kept, ensure_ascii=False, indent=1), encoding="utf-8")
        records = [convert(q) for q in kept]
        for r in records:
            r.check()
        save_jsonl(records, PROCESSED / f"{name}.jsonl")
        print(f"{len(qs)} -> {len(kept)} questions -> {PROCESSED / name}.json, .jsonl")


if __name__ == "__main__":
    main()
