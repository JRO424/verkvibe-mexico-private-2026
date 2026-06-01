#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the 30 Z3 Safety LinkedIn images from the EXACT Gemini prompts
already written in z3_linkedin_content_calendar.csv.

This is the instruction-faithful runner: it does not rewrite or summarize the
prompts. It reads the "Gemini Prompt" column verbatim and sends each one to
Google's image model, saving day-01.png ... day-30.png at 1080x1350 (4:5).

Requirements (auto-installable):
    pip install google-genai pillow

Auth (one required):
    export GEMINI_API_KEY=...        # from Google AI Studio
  or
    export GOOGLE_API_KEY=...

Usage:
    python3 generate_z3_images.py                 # all 30
    python3 generate_z3_images.py --only 6        # just day 6
    python3 generate_z3_images.py --model imagen-3.0-generate-002
"""
import argparse
import csv
import os
import sys
import time

OUT_DIR = "images"
CSV_PATH = "z3_linkedin_content_calendar.csv"
TARGET_W, TARGET_H = 1080, 1350  # 4:5 LinkedIn feed image


def load_prompts():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    items = []
    for i, r in enumerate(rows, 1):
        items.append({
            "day": i,
            "date": r["Date"],
            "prompt": r["Gemini Prompt"],
            "alt": r["Alt Text"],
        })
    return items


def ensure_key():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit(
            "ERROR: no credential found.\n"
            "Set GEMINI_API_KEY (or GOOGLE_API_KEY) to a Google AI Studio key "
            "with image-generation access, then re-run."
        )
    return key


def resize_to_4x5(path):
    """Force the saved image to exactly 1080x1350 (cover-crop, no distortion)."""
    try:
        from PIL import Image
    except ImportError:
        return  # leave native size if Pillow is unavailable
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max(TARGET_W / w, TARGET_H / h)
    img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    w, h = img.size
    left = (w - TARGET_W) // 2
    top = (h - TARGET_H) // 2
    img.crop((left, top, left + TARGET_W, top + TARGET_H)).save(path)


def generate(items, model_name, only=None):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=ensure_key())
    os.makedirs(OUT_DIR, exist_ok=True)
    ok, fail = 0, 0

    for it in items:
        if only and it["day"] != only:
            continue
        out = os.path.join(OUT_DIR, f"day-{it['day']:02d}.png")
        for attempt in range(1, 5):
            try:
                if model_name.startswith("imagen"):
                    resp = client.models.generate_images(
                        model=model_name,
                        prompt=it["prompt"],
                        config=types.GenerateImagesConfig(
                            number_of_images=1,
                            aspect_ratio="3:4",  # closest portrait; cropped to 4:5 after
                        ),
                    )
                    data = resp.generated_images[0].image.image_bytes
                else:
                    resp = client.models.generate_content(
                        model=model_name,
                        contents=it["prompt"],
                        config=types.GenerateContentConfig(
                            response_modalities=["IMAGE"]),
                    )
                    data = None
                    for part in resp.candidates[0].content.parts:
                        if getattr(part, "inline_data", None):
                            data = part.inline_data.data
                            break
                    if data is None:
                        raise RuntimeError("no image part returned")
                with open(out, "wb") as f:
                    f.write(data)
                resize_to_4x5(out)
                print(f"  day {it['day']:02d}  OK  -> {out}")
                ok += 1
                break
            except Exception as e:  # noqa: BLE001
                wait = 2 ** attempt
                print(f"  day {it['day']:02d}  attempt {attempt} failed: {e}")
                if attempt == 4:
                    fail += 1
                else:
                    time.sleep(wait)
    print(f"\nDone. {ok} generated, {fail} failed. Output in ./{OUT_DIR}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="imagen-3.0-generate-002",
                    help="image model id (imagen-3.0-generate-002 or "
                         "gemini-2.5-flash-image)")
    ap.add_argument("--only", type=int, default=None, help="generate one day only")
    args = ap.parse_args()
    generate(load_prompts(), args.model, only=args.only)
