#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a contact sheet and record local source image hashes.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=320)
    args = parser.parse_args()
    if min(args.columns, args.rows, args.tile_size) < 1:
        parser.error("Columns, rows, and tile size must be positive")
    if not args.source.is_dir():
        parser.error(f"Image directory does not exist: {args.source}")
    if args.output.resolve().parent == args.source.resolve():
        parser.error("Write the contact sheet outside the source image directory")
    if args.output.is_dir():
        parser.error("Output must be a file, not a directory")
    try:
        from PIL import Image, ImageOps
    except ImportError:
        parser.error("Pillow is required. Install requirements.txt.")
    files = sorted(path for path in args.source.iterdir()
                   if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    count = args.columns * args.rows
    if len(files) < count:
        parser.error(f"Need {count} images, found {len(files)}")
    selected = files[:count]
    canvas = Image.new("RGB", (args.columns * args.tile_size, args.rows * args.tile_size), "white")
    records = []
    for index, path in enumerate(selected):
        with Image.open(path) as source:
            tile = ImageOps.fit(ImageOps.exif_transpose(source).convert("RGB"),
                                (args.tile_size, args.tile_size), method=Image.Resampling.LANCZOS)
        canvas.paste(tile, (index % args.columns * args.tile_size, index // args.columns * args.tile_size))
        records.append({"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps({
        "selection": "First filenames in lexicographic order; no random sampling",
        "transformation": "EXIF orientation, RGB conversion, center crop and resize to square tiles",
        "columns": args.columns, "rows": args.rows, "tile_size": args.tile_size, "sources": records,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} and its source manifest ({len(selected)} images)")


if __name__ == "__main__":
    main()
