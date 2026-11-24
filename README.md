# Handwriting Font

Turn photos of your handwriting into a real font: TrueType, OpenType and WOFF2.

Write the characters on a sheet of paper, take a photo, and build a font you can install and type
with. Ordinary notebook pages work too. Everything runs on your own computer.

**Landing page:** [`docs/index.html`](docs/index.html) · live at https://yedhu.me/handwriting-font/

## Quick start

```bash
git clone https://github.com/yedhukrishnan/handwriting-font
cd handwriting-font
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
brew install tesseract imagemagick        # Linux: use your package manager
```

1. **Write a sample sheet.** Copy the rows of [`layouts/charsheet.txt`](layouts/charsheet.txt) onto
   paper, one character per square, in a dark pen. Graph paper helps; don't let characters touch.
   Using two pens? Write the whole block twice, and the cleaner copy is chosen per character.
2. **Ingest the photo** (HEIC, JPEG or PNG) into your own project folder:
   ```bash
   .venv/bin/python fontkit.py --project ~/my-font ingest-sheet sheet.jpg
   # sideways photo: --rotate cw|ccw|180   block written twice: --copies 2
   ```
   Check `~/my-font/out/review/sheet_<name>.png`: every glyph is boxed with its label. Rows whose
   glyph count doesn't match the layout are skipped and listed, never guessed.
3. **Build and install:**
   ```bash
   .venv/bin/python fontkit.py --project ~/my-font build                 # out/MyHandwriting.ttf + .woff2
   .venv/bin/python fontkit.py --project ~/my-font build --otf           # out/MyHandwriting.otf
   .venv/bin/python fontkit.py --project ~/my-font build --preset neat   # tidier variant, out/neat/
   ```
   Double-click the `.ttf` to install it. Name your font in `~/my-font/settings.json`.

## Projects

Your handwriting lives in a **project folder**, kept separate from the tool:

```
~/my-font/
├── settings.json        # font name and weights (created from presets/default.json)
├── data/                # cleaned pages, OCR, glyph library, labels, picks
└── out/                 # fonts, specimen.png, review/ images
```

Pass `--project DIR` to every command. Without it, the repository folder itself is the project,
and its `data/`, `out/` and `settings.json` are git-ignored, so nothing personal is committed here.
Keep your project in its own (private) repository if you want it versioned.

## Commands

| Command | What it does |
|---|---|
| `ingest-sheet IMAGE [--copies N] [--rotate cw]` | label a character sample sheet from a layout (`--layout FILE` for your own) |
| `ingest DIR_OR_IMAGES...` | add ordinary notebook pages: clean, OCR, extract letters |
| `region PAGE x0 y0 x1 y1 --tag T` / `label T:k=C` | cut part of a page into numbered pieces and label them by hand |
| `sheet [CHARS]` | contact sheet of candidates per character, best first, with IDs |
| `pick CHAR ID` / `reject ID...` | force or forbid a specific sample |
| `build [--preset neat] [--otf]` | compile the font (+ `.woff2` and `specimen.png`) |
| `reindex` | rebuild the glyph library from saved data |

Quote labels with shell characters: `label 'T:3=?'`.

## Settings and presets

`settings.json` (per project):

| key | meaning | default |
|---|---|---|
| `name` | font family name | `My Handwriting` |
| `lighten` | units shaved off every stroke (per 1000 em) | `20` |
| `cap_width` | horizontal scale for capitals; stroke weight is kept | `0.85` |
| `cap_lighten` | extra thinning for capitals, which tend to be written heavier | `30` |

Presets in [`presets/`](presets) layer on top. `--preset neat` builds "‹name› Neat", a more
consistent variant:

| key | effect |
|---|---|
| `prefer_copy` | `"cleanest"`: take every glyph from the cleanest repetition (pen) of the sheet |
| `stroke_target`, `stroke_strength` | pull each glyph's stroke weight toward the target by thinning/thickening its real outline (proportionally, so thin parts never vanish) |
| `slant_strength`, `slant_cap_deg` | lean glyphs with a stem toward your common slant, capped |
| `contour_smooth` | outline smoothing, as a fraction of glyph height |
| `overshoot`, `optical_sb` | round letters slightly taller / tighter sidebearings |

## How it works

```
photos ─► clean ─► find letters ─► glyph library ─► pick the best ─► build glyph ─► (neat) ─► compile
          OpenCV   sheet layout,    data/library     picks › sheet    fill dry ink,   one pen,   fontTools,
                   OCR, hand        + index.json     › labels › OCR;  smooth along    even       OTS-validated
                   labels                            cleanest pen     the outline     weight &
                                                                                      slant
```

1. **Clean.** Lighting is flattened against each page's own background and only dark ink is kept.
   Grid lines, yellow highlighter, red underlines and page shadows are removed.
2. **Find letters.** Sample sheets are segmented into rows and matched to the layout. Notebook pages
   go through Tesseract: when a word's ink blobs match its letter count, each blob gets its letter.
   Anything else can be labelled by hand from page regions.
3. **Glyph library.** Every labelled letter is saved as an image, with its position relative to the
   baseline. Re-running only adds; samples whose source page isn't present are kept.
4. **Pick the best.** Order: your picks › sheet › hand labels › OCR. Between pens, specks, pin-holes
   and ragged edges are scored and the cleanest wins. Ties go to the sample closest to the median
   shape of all samples (your everyday writing).
5. **Build glyph.** Thin dry-ink holes and cracks are filled (real counters in o, e, a… are kept),
   the outline is smoothed along the contour (never blurred, so close strokes don't merge), lightened,
   capitals condensed, and the result is scaled to x-height / cap height / descender. Symbols get
   typographic positions; `:` and `;` are built from your `.` and `,`.
6. **Compile.** fontTools writes the font with full naming tables, overlap removal and a `.notdef`
   glyph; the OpenType Sanitizer accepts the result.

## Privacy

- Processing is local; nothing is uploaded.
- HEIC photos are converted with metadata stripped.
- Your project folder is yours: the tool never writes personal data into this repository.
