#!/usr/bin/env python3
"""Render text in one or more fonts to a PNG so you can look at it.

Usage:
  render_text.py OUT.png --font A.ttf [--font B.ttf ...] [--size 56] [LINE ...]

With several fonts, each line is drawn once per font, labelled with the font's
file name, so versions can be compared side by side. Without LINEs a default
specimen (alphabet, digits, symbols, pangrams) is used.
"""
import argparse, os
from PIL import Image, ImageDraw, ImageFont

DEFAULT = [
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
    'abcdefghijklmnopqrstuvwxyz',
    '0123456789 .,:;!?\'"-()[]{}/\\|@#$%&*+=^~`',
    'The quick brown fox jumps over the lazy dog.',
    'SPHINX OF BLACK QUARTZ, JUDGE MY VOW - 24/11/2026',
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--font', action='append', required=True)
    ap.add_argument('--size', type=int, default=56)
    ap.add_argument('lines', nargs='*')
    a = ap.parse_args()
    lines = a.lines or DEFAULT
    fonts = [(os.path.basename(p), ImageFont.truetype(p, a.size)) for p in a.font]
    try:
        label = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 18)
    except OSError:
        label = ImageFont.load_default()
    gutter = (max(int(label.getlength(n)) for n, _ in fonts) + 30) if len(fonts) > 1 else 20
    row_h = int(a.size * 1.45)
    width = gutter + max(int(f.getlength(t)) for _, f in fonts for t in lines) + 40
    height = 20 + len(lines) * (len(fonts) * row_h + 18)
    img = Image.new('L', (min(width, 4000), height), 255)
    d = ImageDraw.Draw(img)
    y = 10
    for t in lines:
        for name, f in fonts:
            if len(fonts) > 1:
                d.text((10, y + a.size * 0.35), name, font=label, fill=120)
            d.text((gutter, y), t, font=f, fill=0)
            y += row_h
        y += 18
    img.save(a.out)
    print(a.out)

if __name__ == '__main__':
    main()
