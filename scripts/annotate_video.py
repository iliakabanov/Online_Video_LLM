"""Видео CLEVRER-Proactive для презентации: номер кадра и состояние вопросов поверх кадров.

Вопросы задаются id записей одного видео (порядок id — номера Q1, Q2, …). Вопрос показывается плашкой внизу: до
кадра, с которого ответ виден (answers[0].start_frame), — «Q1 …»; начиная с него и до конца видео — «Q1 ✓ <ответ>».
Первые --flash секунд после события плашка выделена, чтобы момент было видно.

python scripts/annotate_video.py --ids validation_10000_clevrer_query_shape_0 validation_10000_first_hit_6 \
    --output outputs/presentation/video_10000.mp4 [--speed 0.5] [--scale 2]
"""
import argparse
import json
import os
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
FONTS = ("DejaVuSansMono-Bold.ttf", "DejaVuSans-Bold.ttf", "consolab.ttf", "arialbd.ttf")
INK, MUTED, PANEL, FLASH = (255, 255, 255), (170, 170, 170), (0, 0, 0, 165), (255, 214, 0)


def font(size: int):
    for name in FONTS:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def load_records(path: Path, ids: list[str]) -> list[dict]:
    by_id = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["id"] in ids:
                by_id[record["id"]] = record
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"Нет записей {missing} в {path}")
    records = [by_id[i] for i in ids]
    if len({r["video"] for r in records}) != 1:
        raise SystemExit("Все вопросы должны быть к одному видео")
    return records


def check_mark(draw: ImageDraw.ImageDraw, x: float, y: float, size: float, color, width: int):
    """Галочка линиями: в шрифтах по умолчанию символа ✓ часто нет."""
    draw.line([(x, y + size * 0.55), (x + size * 0.38, y + size * 0.9), (x + size, y + size * 0.1)], fill=color,
              width=width, joint="curve")


def annotate(frame: np.ndarray, index: int, total: int, fps: float, questions: list[dict], flash_frames: int,
             scale: int) -> np.ndarray:
    image = Image.fromarray(frame).resize((frame.shape[1] * scale, frame.shape[0] * scale), Image.LANCZOS)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    pad, big = 10 * scale, font(15 * scale)

    text = f"кадр {index:03d} / {total - 1:03d}   {index / fps:4.2f} с".replace(".", ",")
    box = draw.textbbox((0, 0), text, font=big)
    draw.rounded_rectangle([pad, pad, pad * 3 + box[2], pad * 2.6 + box[3]], radius=6 * scale, fill=PANEL)
    draw.text((pad * 2, pad * 1.8), text, font=big, fill=INK)

    x, y = pad, image.height - pad
    for q in questions:
        known = index >= q["event"]
        label = f"{q['name']}  {q['answer']}" if known else f"{q['name']}  …"
        box = draw.textbbox((0, 0), label, font=big)
        mark = box[3] if known else 0
        width, height = box[2] + mark + pad * (3 if known else 2), box[3] + pad * 1.6
        top = y - height
        flashing = known and index - q["event"] < flash_frames
        draw.rounded_rectangle([x, top, x + width, y], radius=6 * scale, fill=PANEL,
                               outline=FLASH if flashing else None, width=3 * scale)
        color = INK if known else MUTED
        draw.text((x + pad, top + pad * 0.6), q["name"], font=big, fill=color)
        if known:
            name_width = draw.textlength(q["name"] + "  ", font=big)
            check_mark(draw, x + pad + name_width, top + pad * 0.8, box[3] * 0.85, FLASH if flashing else INK,
                       max(2, 2 * scale))
            draw.text((x + pad * 2 + name_width + mark, top + pad * 0.6), q["answer"], font=big, fill=color)
        else:
            draw.text((x + pad + draw.textlength(q["name"] + "  ", font=big), top + pad * 0.6), "…", font=big,
                      fill=color)
        x += width + pad
    return np.asarray(Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", nargs="+", required=True, help="id вопросов одного видео; порядок — Q1, Q2, …")
    parser.add_argument("--output", required=True, help="mp4")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", str(REPO / "data")))
    parser.add_argument("--records", default="processed/clevrer_proactive/val_group1.jsonl",
                        help="записи вопросов относительно --data-root")
    parser.add_argument("--speed", type=float, default=0.5, help="скорость воспроизведения (0.5 — вдвое медленнее)")
    parser.add_argument("--scale", type=int, default=2, help="увеличение кадра, чтобы надписи были чёткими")
    parser.add_argument("--flash", type=float, default=0.5, help="сколько секунд видео выделять плашку после события")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    records = load_records(data_root / args.records, args.ids)
    fps = float(records[0]["fps"])
    questions = [{"name": f"Q{i + 1}", "event": r["answers"][0]["start_frame"], "answer": r["answers"][0]["answer"]}
                 for i, r in enumerate(records)]
    for q, r in zip(questions, records):
        print(f"{q['name']}: {r['question']} → {q['answer']} с кадра {q['event']}")

    video = data_root / records[0]["video"]
    reader = imageio_ffmpeg.read_frames(str(video))
    meta = next(reader)
    width, height = meta["size"]
    frames = [np.frombuffer(f, dtype=np.uint8).reshape(height, width, 3) for f in reader]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(str(output), (width * args.scale, height * args.scale), fps=fps * args.speed,
                                         codec="libx264", pix_fmt_out="yuv420p", quality=8, macro_block_size=1)
    writer.send(None)
    for index, frame in enumerate(frames):
        writer.send(np.ascontiguousarray(annotate(frame, index, len(frames), fps, questions,
                                                  round(args.flash * fps), args.scale)))
    writer.close()
    print(f"{len(frames)} кадров, {len(frames) / (fps * args.speed):.1f} с → {output}")


if __name__ == "__main__":
    main()
