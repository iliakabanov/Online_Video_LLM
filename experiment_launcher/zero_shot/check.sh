#!/usr/bin/env bash
# Сверка перед полным прогоном (src/online_vllm/eval/ZERO_SHOT.md, «Запуск на сервере», шаг 3): обе системы MMDuet2 на
# проверочном вопросе validation_10000_clevrer_query_material_1 («What material is the object to collide with the
# sphere?», ответ виден с кадра 84 = пара 42). Печатает P(NO REPLY) онлайн-модели по парам и ответы обеих систем.
# Ожидаемая картина (на Apple M4, transformers 5.3): провалы P(NO REPLY) до события — около пар 8–12 (въезд куба,
# минимум ~0,3) и 17–20 (первое столкновение, ~0,5); с пары 24 и после события P ≈ 1. Точные значения на другом
# GPU могут отличаться (в отдельных парах до ~0,3–0,4), положение провалов — нет.
#   DATA_ROOT=/data bash experiment_launcher/zero_shot/check.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/PATH/TO/DATA}"
DEVICE="${DEVICE:-cuda:0}"
OUT="${OUT:-$ROOT/results/zero_shot_check}"
ID="validation_10000_clevrer_query_material_1"
[[ "$DATA_ROOT" != "/PATH/TO/DATA" ]] || { echo "Задайте DATA_ROOT." >&2; exit 2; }
export DATA_ROOT
cd "$ROOT"

rm -rf "$OUT"  # иначе zero_shot_probe.py пропустит уже посчитанный вопрос
for system in mmduet2_online mmduet2_offline; do
  python scripts/zero_shot_probe.py --system "$system" --device "$DEVICE" --ids "$ID" \
    --questions processed/clevrer_proactive/val_group1.jsonl --output-dir "$OUT/$system"
done

OUT="$OUT" python - <<'PY'
import json, math, os
from pathlib import Path
out = Path(os.environ["OUT"])
row = lambda system: json.loads((out / system / "answers.jsonl").read_text(encoding="utf-8").splitlines()[0])
online, offline = row("mmduet2_online"), row("mmduet2_offline")
quiet = [math.exp(v) for v in online["silence"]]
target = online["pairs"]["target"]
print(f"\nP(NO REPLY) по парам (ответ виден с пары {target}):")
for start in range(0, 64, 16):
    print(f"  {start:2d}–{start + 15:2d}: " + " ".join(f"{p:.2f}" for p in quiet[start:start + 16]))
before = min(range(target), key=quiet.__getitem__)
after = sorted(quiet[target:])[len(quiet[target:]) // 2]
print(f"минимум до события: {quiet[before]:.2f} на паре {before}; медиана с пары {target}: {after:.2f}")
first_letter = lambda r, pair: (lambda e: e[0] / sum(e))([math.exp(v) for v in r["letters"][pair]])
for name, r in (("онлайн", online), ("офлайн", offline)):
    print(f"P(A = rubber, верный) среди A/B, {name}: " + ", ".join(
        f"{c} {first_letter(r, p):.2f}" for c, p in r["pairs"].items()))
ok = quiet[before] < 0.8 and after > 0.95
print("\nКартина как ожидалась." if ok else "\nКартина отличается от ожидаемой: сверьте с ZERO_SHOT.md, прежде чем считать всё.")
PY
