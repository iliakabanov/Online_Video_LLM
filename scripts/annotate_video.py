"""Видео CLEVRER-Proactive для презентации: номер кадра и состояние вопросов поверх кадров.

Вопросы задаются id записей одного видео (порядок id — номера Q1, Q2, …). Вопрос показывается плашкой внизу: до
кадра, с которого ответ виден (answers[0].start_frame), — «Q1 …»; начиная с него и до конца видео — «Q1 ✓ <ответ>».
Первые --flash секунд после события плашка выделена, чтобы момент было видно.

python scripts/annotate_video.py --ids validation_10000_clevrer_query_shape_0 validation_10000_first_hit_6 \
    --output outputs/presentation/video_10000.mp4 [--speed 0.5] [--scale 2]

С --answers (JSON scripts/answer_video.py) вместо этого показывается работа головы готовности: у каждого вопроса
график P(готов) от начала видео до текущего кадра с порогом τ (пунктир) и текущее значение. На кадре видна P
последней пары, закончившейся на нём или раньше, — то, что голова уже могла посчитать. Когда P впервые превышает τ,
плашка вспыхивает и до конца видео показывает ответ модели (✓ — верный, ✗ — неверный). Засечка на графике —
кадр, с которого ответ виден в видео; она появляется, когда видео до него доходит.

python scripts/annotate_video.py --answers outputs/presentation/video_10000_answers.json \
    --output outputs/presentation/video_10000_readiness.mp4 [--lang en]
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
WRONG = (255, 110, 90)
TEXT = {"ru": {"frame": "кадр", "seconds": "с", "comma": True},
        "en": {"frame": "frame", "seconds": "s", "comma": False}}


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


def frame_counter(draw: ImageDraw.ImageDraw, index: int, total: int, fps: float, scale: int, lang: str = "ru"):
    pad, big, text = 10 * scale, font(15 * scale), TEXT[lang]
    label = f"{text['frame']} {index:03d} / {total - 1:03d}   {index / fps:4.2f} {text['seconds']}"
    label = label.replace(".", ",") if text["comma"] else label
    box = draw.textbbox((0, 0), label, font=big)
    draw.rounded_rectangle([pad, pad, pad * 3 + box[2], pad * 2.6 + box[3]], radius=6 * scale, fill=PANEL)
    draw.text((pad * 2, pad * 1.8), label, font=big, fill=INK)


def annotate(frame: np.ndarray, index: int, total: int, fps: float, questions: list[dict], flash_frames: int,
             scale: int) -> np.ndarray:
    image = Image.fromarray(frame).resize((frame.shape[1] * scale, frame.shape[0] * scale), Image.LANCZOS)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    pad, big = 10 * scale, font(15 * scale)
    frame_counter(draw, index, total, fps, scale)

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


def readiness_questions(answers: dict, total: int) -> list[dict]:
    """Из JSON answer_video.py: по каждому вопросу P(готов) и ответ на каждом кадре — решение последней пары,
    закончившейся на кадре или раньше (None до конца первой пары)."""
    questions = []
    for i, q in enumerate(answers["questions"]):
        pairs = sorted(q["per_pair"], key=lambda p: p["frame"])
        per_frame, k = [], -1
        for frame in range(total):
            while k + 1 < len(pairs) and pairs[k + 1]["frame"] <= frame:
                k += 1
            per_frame.append(pairs[k]["p_ready"] if k >= 0 else None)
        questions.append({"name": f"Q{i + 1}", "p": per_frame, "event": q["answer_visible_frame"],
                          "fired_frame": q["answer_frame"] if q["fired"] else None, "answer": q["answer"],
                          "correct": q["correct"]})
    return questions


def dashed(draw: ImageDraw.ImageDraw, x0: float, x1: float, y: float, color, width: int, dash: float):
    x = x0
    while x < x1:
        draw.line([(x, y), (min(x + dash, x1), y)], fill=color, width=width)
        x += dash * 2


def annotate_readiness(frame: np.ndarray, index: int, total: int, fps: float, questions: list[dict],
                       threshold: float, flash_frames: int, scale: int, lang: str) -> np.ndarray:
    image = Image.fromarray(frame).resize((frame.shape[1] * scale, frame.shape[0] * scale), Image.LANCZOS)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    pad, big = 10 * scale, font(15 * scale)
    frame_counter(draw, index, total, fps, scale, lang)

    chart_w, chart_h = 150 * scale, 34 * scale
    y = image.height - pad
    for q in reversed(questions):  # снизу вверх: Q1 сверху
        p = q["p"][index]
        fired = q["fired_frame"] is not None and index >= q["fired_frame"]
        flashing = fired and index - q["fired_frame"] < flash_frames
        name_w = draw.textlength(q["name"], font=big)
        value = "—" if p is None else f"P {p:.2f}"
        value_w = draw.textlength("P 0.00", font=big)
        answer_w = draw.textlength(q["answer"], font=big) + 1.6 * pad if fired else 0
        width = pad + name_w + pad + chart_w + pad + value_w + (pad + answer_w if fired else 0) + pad
        top = y - chart_h - pad
        draw.rounded_rectangle([pad, top, pad + width, y], radius=6 * scale, fill=PANEL,
                               outline=FLASH if flashing else None, width=3 * scale)
        x = pad * 2
        draw.text((x, top + (chart_h + pad) / 2), q["name"], font=big, fill=INK, anchor="lm")
        x += name_w + pad

        # График P(готов): ось 0..1 по высоте, кадры 0..total-1 по ширине; порог пунктиром.
        cx0, cy0, cy1 = x, top + pad / 2, top + pad / 2 + chart_h
        fx = lambda f: cx0 + chart_w * f / (total - 1)  # noqa: E731
        fy = lambda v: cy1 - (cy1 - cy0) * v  # noqa: E731
        draw.rectangle([cx0, cy0, cx0 + chart_w, cy1], fill=(255, 255, 255, 25))
        dashed(draw, cx0, cx0 + chart_w, fy(threshold), MUTED, max(1, scale), 4 * scale)
        if index >= q["event"]:
            draw.line([(fx(q["event"]), cy0), (fx(q["event"]), cy1)], fill=MUTED, width=max(1, scale))
        points = [(fx(f), fy(v)) for f, v in enumerate(q["p"][:index + 1]) if v is not None]
        if len(points) > 1:
            draw.line(points, fill=FLASH if fired else INK, width=2 * scale, joint="curve")
        if points:
            r = 3 * scale
            draw.ellipse([points[-1][0] - r, points[-1][1] - r, points[-1][0] + r, points[-1][1] + r],
                         fill=FLASH if fired else INK)
        x += chart_w + pad

        draw.text((x, top + (chart_h + pad) / 2), value, font=big, fill=FLASH if fired else INK, anchor="lm")
        x += value_w + pad
        if fired:
            mark, color = chart_h * 0.45, (INK if q["correct"] else WRONG)
            my = top + (chart_h + pad) / 2 - mark / 2
            if q["correct"]:
                check_mark(draw, x, my, mark, FLASH if flashing else INK, max(2, 2 * scale))
            else:
                draw.line([(x, my), (x + mark, my + mark)], fill=WRONG, width=max(2, 2 * scale))
                draw.line([(x, my + mark), (x + mark, my)], fill=WRONG, width=max(2, 2 * scale))
            draw.text((x + mark + 0.6 * pad, top + (chart_h + pad) / 2), q["answer"], font=big, fill=color,
                      anchor="lm")
        y = top - pad / 2
    return np.asarray(Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", nargs="+", help="id вопросов одного видео; порядок — Q1, Q2, …")
    parser.add_argument("--answers", help="JSON scripts/answer_video.py: показать P(готов) и ответы головы")
    parser.add_argument("--lang", choices=tuple(TEXT), default="ru", help="язык подписей (с --answers)")
    parser.add_argument("--output", required=True, help="mp4")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", str(REPO / "data")))
    parser.add_argument("--records", default="processed/clevrer_proactive/val_group1.jsonl",
                        help="записи вопросов относительно --data-root")
    parser.add_argument("--speed", type=float, default=0.5, help="скорость воспроизведения (0.5 — вдвое медленнее)")
    parser.add_argument("--scale", type=int, default=2, help="увеличение кадра, чтобы надписи были чёткими")
    parser.add_argument("--flash", type=float, default=0.5, help="сколько секунд видео выделять плашку после события")
    args = parser.parse_args()
    answers = json.loads(Path(args.answers).read_text(encoding="utf-8")) if args.answers else None
    ids = [q["id"] for q in answers["questions"]] if answers else args.ids
    if not ids:
        parser.error("нужен --ids или --answers")

    data_root = Path(args.data_root)
    records = load_records(data_root / args.records, ids)
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
    flash = round(args.flash * fps)
    if answers:
        readiness = readiness_questions(answers, len(frames))
        for q in readiness:
            print(f"{q['name']}: P > τ = {answers['threshold']} с кадра {q['fired_frame']}, ответ {q['answer']} "
                  f"({'верно' if q['correct'] else 'неверно'})")
    for index, frame in enumerate(frames):
        if answers:
            annotated = annotate_readiness(frame, index, len(frames), fps, readiness, answers["threshold"], flash,
                                           args.scale, args.lang)
        else:
            annotated = annotate(frame, index, len(frames), fps, questions, flash, args.scale)
        writer.send(np.ascontiguousarray(annotated))
    writer.close()
    print(f"{len(frames)} кадров, {len(frames) / (fps * args.speed):.1f} с → {output}")


if __name__ == "__main__":
    main()
