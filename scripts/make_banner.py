#!/usr/bin/env python3
"""Build the README banner: one archived FLUX.1-dev sample per poem, Imagist row above Romantic row.

Tiles are cut from the saved 5x5 contact sheets in assets/ (one poem per row, five samples per row).
No image is generated or edited beyond cropping, resizing and rounding the corners.
assets/banner.png was built on macOS with Helvetica; elsewhere another font is used for the two
labels, so only the label pixels differ.
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SHEETS = (("Imagism", ROOT / "assets/flux-imagism.jpg"), ("Romanticism", ROOT / "assets/flux-romanticism.jpg"))
FONTS = ("/System/Library/Fonts/Helvetica.ttc", "DejaVuSans.ttf", "Arial.ttf")
GRID, TILE, GAP, PAD, LABEL_H, FONT_SIZE = 5, 300, 10, 12, 44, 26


def tiles(sheet: Path, column: int):
    with Image.open(sheet) as source:
        im = source.convert("RGB")
    if im.width != im.height:
        raise ValueError(f"{sheet}: expected a square {GRID}x{GRID} sheet, got {im.width}x{im.height}")
    step = im.width // GRID
    for row in range(GRID):
        box = (column * step, row * step, (column + 1) * step, (row + 1) * step)
        yield im.crop(box).resize((TILE, TILE), Image.Resampling.LANCZOS)


def rounded(im: Image.Image, radius: int = 14) -> Image.Image:
    mask = Image.new("L", (im.width * 4, im.height * 4), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, mask.width - 1, mask.height - 1), radius * 4, fill=255)
    out = im.convert("RGBA")
    out.putalpha(mask.resize(im.size, Image.Resampling.LANCZOS))
    return out


def label_font():
    for name in FONTS:
        try:
            return ImageFont.truetype(name, FONT_SIZE)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=FONT_SIZE)
    except TypeError:
        return ImageFont.load_default()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--column", type=int, default=1,
                    help=f"which sample (0 to {GRID - 1}) to take from each poem row; default 1")
    ap.add_argument("--output", type=Path, default=ROOT / "assets/banner.png")
    args = ap.parse_args()
    if not 0 <= args.column < GRID:
        ap.error(f"--column must be between 0 and {GRID - 1}")

    width = PAD * 2 + GRID * TILE + (GRID - 1) * GAP
    height = PAD * 2 + 2 * (LABEL_H + TILE) + GAP
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    font = label_font()

    y = PAD
    try:
        for label, sheet in SHEETS:
            draw.text((PAD + 2, y + 8), label, fill=(139, 148, 158, 255), font=font)
            y += LABEL_H
            for i, tile in enumerate(tiles(sheet, args.column)):
                canvas.alpha_composite(rounded(tile), (PAD + i * (TILE + GAP), y))
            y += TILE + GAP
    except (OSError, ValueError) as error:
        ap.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, optimize=True)
    print(args.output, canvas.size)


if __name__ == "__main__":
    main()
