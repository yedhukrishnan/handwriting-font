#!/usr/bin/env python3
"""Measure how consistent a built font is, and flag glyphs that look broken.

Usage (run with the repo's venv python from the repo root):
  .venv/bin/python .claude/skills/make-handwriting-font/scripts/font_report.py FONT.ttf [OTHER.ttf ...]

Prints, per font:
  - stroke weight: median (font units per 1000 em) and variation between glyphs (CV %)
  - capitals vs lowercase stroke weight (capitals are usually written heavier)
  - slant spread between glyphs with a vertical stem (std dev, degrees)
  - glyphs that are suspiciously heavy or light compared with the rest (> 30% off median)
  - with two fonts: glyphs whose number of separate pieces differs (a stroke broke or merged)
"""
import os, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
sys.argv_backup, sys.argv = sys.argv, ['fontkit']
import fontkit as fk          # reuse stroke_px / estimate_slant / SLANT_SET
sys.argv = sys.argv_backup

LETTERS = 'ABCDEFGHJKLMNOPRSTUVWXYZabcdefghkmnoprstuvwxyz0123456789'

def raster(font, c, size=600):
    im = Image.new('L', (int(size * 1.6), int(size * 1.6)), 0)
    ImageDraw.Draw(im).text((size // 6, size // 6), c, font=font, fill=255)
    return (np.array(im) > 127).astype(np.uint8)

def report(path):
    f = ImageFont.truetype(path, 600)
    have = {chr(u) for u in TTFont(path).getBestCmap()}
    sw, sl = {}, []
    for c in LETTERS:
        if c not in have: continue
        a = raster(f, c)
        sw[c] = fk.stroke_px(a) / 600 * 1000
        if c in fk.SLANT_SET: sl.append(fk.estimate_slant(a))
    v = np.array(list(sw.values())); med = float(np.median(v))
    caps = [sw[c] for c in sw if c.isupper()]; lower = [sw[c] for c in sw if c.islower()]
    print(f'== {os.path.basename(path)}')
    print(f'stroke weight: median {med:.0f} units, variation {v.std() / v.mean() * 100:.0f}%')
    if caps and lower:
        print(f'capitals vs lowercase: {np.median(caps):.0f} vs {np.median(lower):.0f}')
    if sl: print(f'slant spread: {np.std(sl):.1f} deg (median {np.median(sl):+.0f} deg)')
    odd = sorted((c for c in sw if abs(sw[c] - med) > 0.3 * med), key=lambda c: -abs(sw[c] - med))
    print('outlier weights:', ', '.join(f'{c}={sw[c]:.0f}' for c in odd) or 'none')
    return f, have

def pieces(font, c):
    return cv2.connectedComponents(raster(font, c, 300), connectivity=8)[0] - 1

if __name__ == '__main__':
    paths = sys.argv[1:]
    if not paths: sys.exit(__doc__)
    loaded = [report(p) for p in paths]
    if len(loaded) == 2:
        (fa, ha), (fb, hb) = loaded
        diff = [(c, pieces(fa, c), pieces(fb, c)) for c in sorted(ha & hb) if c.isprintable() and not c.isspace()
                and pieces(fa, c) != pieces(fb, c)]
        print('pieces differ (char, first, second):', diff or 'none')
