# Fixing glyphs

How fontkit chooses a sample for each character, and the recipes for problems that come up.

## How a sample is chosen

For each character, candidates are ordered:
1. your `pick` (always wins, even over the quality filter)
2. sample-sheet glyphs: when the sheet was written more than once, only the copies with
   the fewest defects (specks + pin-holes + ragged edges) compete
3. hand-labelled regions (`region`/`label`)
4. OCR-extracted glyphs from notebook pages

Ties go to the sample closest to the median shape of *all* samples of that character,
which is the version most like the person's everyday writing.

## Contact sheets

```bash
.venv/bin/python fontkit.py --project P sheet 'aAgG?'      # quote shell characters
```
Read `P/out/review/sheet.png` (use `--out name.png` to keep several). Each row is one
character, best first (copies the quality filter set aside are shown last, so a user who
prefers them can still pick them); the text under each thumbnail is the end of the sample ID. Full
IDs: `m:<tag>:<k>` (hand-labelled), `s:<sheet>:<row>:<col>` (sample sheet),
`<page>@x,y` (OCR). Get exact IDs with:
```bash
.venv/bin/python -c "import sys; sys.argv=['x']; import fontkit as k; k.set_project('P'); print([s['id'] for s in k.ranked(keep_all=True)['g']])"
```
Then `pick g <id>` or `reject <id> ...` and rebuild.

## Recipes

**Letter merged with a neighbour or a stray mark** (contact sheet shows two shapes):
`reject` that ID; for a sheet, rewrite it with more space.

**Hollow / dotted / bitten strokes** (faint or drying pen): usually another copy is
cleaner. `pick` it. Thin enclosed gaps are filled automatically. Characters that should
never have holes are filled completely; others (in `COUNTERS`: a b d e g o p q A B D O P
Q R 0 4 6 8 9 & @ % # ?) keep holes wider than ~0.22 × stroke width.

**A gap that should be open got closed** (e.g. the top of an S touching its body): look
at the raw sample in `P/data/library/U+XXXX/`. If the gap exists there, it's the
processing; try a different sample or report it. If not, the writing itself touches.

**Letter is a different character** (OCR mislabel): `reject` it. Mislabels are common for
o/O, s/S, c/C, x/X, v/V, w/W, l/I/1, and for capitals and digits generally.

**Capitals too wide or heavy:** in `P/settings.json`, lower `cap_width` (0.85 → 0.8) and/or
raise `cap_lighten` (units removed from capital strokes). Check with `font_report.py`;
capitals within ~10% of lowercase weight look balanced.

**Punctuation at the wrong height:** symbol positions are typographic (`SYMBOL_POS` in
fontkit.py), not measured, because hand-written symbol rows rarely share a baseline.
Adjust there only if something is clearly off for everyone.

**Neat variant issues:** `--preset neat` thins/thickens outlines toward a common weight and
shears stemmed letters toward the common slant. Compare the two builds with
`font_report.py REGULAR.ttf NEAT.ttf`. The "pieces differ" line catches strokes that broke
or crumbs that appeared. To make one letter use a different sample in Neat, `pick` it
(picks apply to both builds).

## Settings reference (`P/settings.json`)

| key | effect | default |
|---|---|---|
| `name` | font family name | My Handwriting |
| `lighten` | units shaved off every stroke | 20 |
| `cap_width` | horizontal scale for capitals (stroke weight kept) | 0.85 |
| `cap_lighten` | extra thinning for capitals | 30 |

Preset `presets/neat.json` keys: `prefer_copy` ("cleanest" or a copy number),
`stroke_target`, `stroke_strength`, `slant_strength`, `slant_cap_deg`, `contour_smooth`,
`overshoot`, `optical_sb`. A project can carry its own preset file: `--preset path.json`.
