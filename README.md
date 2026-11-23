# handwriting-font

Builds a TrueType/OpenType font from photos of handwritten notebook pages.
Every intermediate step is saved in `data/`. Adding more pages makes the font better without redoing anything.

```
.venv/bin/python fontkit.py build          # -> out/Yedhu.ttf + out/specimen.png
.venv/bin/python fontkit.py build --otf    # -> out/Yedhu.otf
.venv/bin/python fontkit.py build --name "Yedhu Bold"
```

(First time on a new machine: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`; also needs `tesseract` and `imagemagick` from Homebrew.)

## How it works

1. **ingest**: each photo is binarized. Lighting is flattened, and grid lines, yellow highlighter and red underlines are removed. The result goes to `data/pages/`. Tesseract OCR output goes to `data/ocr/`.
2. **auto-extract**: for each OCR word whose number of ink blobs equals its number of letters, each blob becomes a labeled sample in `data/library/U+XXXX/`. All samples are listed in `data/index.json`.
3. **manual labels**: capitals, digits and punctuation are rare or OCR'd badly, so they are labeled by hand from page regions (`data/regions.json` + `data/manual_labels.json`).
4. **selection**: per character, the order is `pick` (yours), then manual labels, then the auto sample closest to the median shape. Rejected IDs are skipped (`data/selections.json`).
5. **build**: the chosen sample is smoothed and vectorized, scaled to its class height (x-height / cap / ascender / descender), and written with fontTools. `i`/`j` get a synthetic dot. `Q`, `.`, `,`, `:`, `;`, `-`, `'`, `"` and `!` are synthesized where no sample exists.

## Improving the font with more handwriting

### Option A: just add more notebook pages (easiest)

```
.venv/bin/python fontkit.py ingest "/path/to/folder/of/photos"     # or individual image files
.venv/bin/python fontkit.py build
```

More pages give more lowercase samples, and the median-based pick gets more reliable.
Pages that were already processed are skipped (`--force` redoes them).

### Option B: write a "sample sheet" (best quality, fills the gaps)

Characters that are currently weak or missing:

- **Weak:** `8`, `x`, `4`, `K`, `Q` (synthesized), `!` (synthesized), quotes/comma (synthesized)
- **Missing:** `# $ % & ( ) * + < = > @ [ \ ] ^ _ { | } ~`

Write these on a **plain or lightly-gridded** page in the same pen, with each character
**separate and not touching anything** (no underlines, boxes or highlighter over them). Good layout:

```
a b c d e f g h i j k l m n o p q r s t u v w x y z
A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
0 1 2 3 4 5 6 7 8 9
. , ; : ! ? ' " - ( ) [ ] / & @ # % + = *
```

Photograph it flat and evenly lit, then:

```
.venv/bin/python fontkit.py ingest sheet.jpg
# OCR auto-labels most of it. For anything it misses, cut the line into numbered pieces:
.venv/bin/python fontkit.py region sheet 0 1200 3000 1400 --tag digits   # page name, x0 y0 x1 y1 (full-res px)
#   -> look at out/review/region_digits.png, then label the numbered pieces:
.venv/bin/python fontkit.py label digits:0=0 digits:1=1 digits:7=8 'digits:12=&'
.venv/bin/python fontkit.py build
```

(Quote labels that contain shell characters such as `?`, `*`, `&`, `(`.)
`region` also takes `--rot cw|ccw` for sideways text.

### Reviewing and correcting choices

```
.venv/bin/python fontkit.py sheet abcxyz        # -> out/review/sheet.png, best-first, ID under each sample
.venv/bin/python fontkit.py pick g IMG_20250320_010300582@3493,944
.venv/bin/python fontkit.py reject IMG_xxx@123,456 m:right:31
```

## Layout

| Path | Contents |
|---|---|
| `data/sources.json` | page name → original photo path |
| `data/pages/` | binarized pages |
| `data/ocr/` | Tesseract TSVs |
| `data/library/` | every labeled glyph sample (PNG), rebuilt by `reindex` |
| `data/index.json` | sample metadata |
| `data/regions.json`, `data/manual_labels.json` | hand-labeled regions (re-extracted from pages, reproducible) |
| `data/selections.json` | your picks / rejects |
| `out/` | font files, specimen, review sheets |

Glyph metrics (x-height 420, cap 640, ascender 690, descender −250 per 1000 em) and the
side-bearing (40) are constants near the top of the font-build section in `fontkit.py`.

## Stroke weight & clean-up

Glyphs keep their natural handwritten outline. Dry-ink artefacts are repaired before
vectorizing: thin enclosed holes and cracks are filled. Real counters (o, e, a, 6 …) are kept
because they are wider than ~0.22× the pen stroke, and only characters listed in `COUNTERS`
may have holes at all.

The weight is adjusted by evenly thinning every edge, which keeps the pen's natural thick/thin:

```
.venv/bin/python fontkit.py build                 # default: 20 units lighter than the ink
.venv/bin/python fontkit.py build --lighten 0     # original ink weight
.venv/bin/python fontkit.py build --lighten 35    # lighter still
```

(`--monoline --weight N` redraws every glyph with one uniform pen. It is even, but loses the hand-written character, so it's off by default.)

The build also runs overlap removal and writes full name-table entries (family, full name,
PostScript name, version, unique ID) and a `.notdef` box, so validators such as macOS Font Book accept it.

## What's in this repository

The original journal photos and everything that reproduces their text are **not** committed
(see `.gitignore`): `data/pages/`, `data/ocr/`, `data/sources.json`, `out/review/`, and
`out/journal.html` stay local. The repo contains the code, the extracted per-letter glyph
library (`data/library/` + `data/index.json`), the manual labels/selections, and the built fonts.

`fontkit.py reindex`, `label` and `build` work on a fresh clone. Samples whose source page isn't
present locally are carried over from the committed library rather than dropped, so you can keep
improving the font by ingesting new pages.
