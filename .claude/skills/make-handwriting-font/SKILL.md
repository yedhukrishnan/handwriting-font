---
name: make-handwriting-font
description: Build, review and improve a font (TTF/OTF/WOFF2) from photos of someone's handwriting using this repo's fontkit.py. Use whenever the user wants to turn their handwriting, a character sample sheet, or photographed notebook/journal pages into a font, or to fix how their handwriting font looks (a letter looks wrong, broken, blobby, too wide/heavy/light, misplaced punctuation, missing characters), make a tidier "neat" variant, add more handwriting samples, or install the font — even if they don't mention fontkit by name.
---

# Make a handwriting font

`fontkit.py` (repo root) does the deterministic work: cleaning photos, cutting out letters,
vectorizing, compiling fonts. Your job is the judgment it can't do: checking that every
letter is labelled right, looking at the result, and choosing fixes. **Always look at the
images the tool produces** (with the Read tool) before reporting success. Most problems
are only visible, not in the command output.

Run everything from the repo root with the repo's venv:
`.venv/bin/python fontkit.py --project <DIR> <command>`.

## Ground rules (why they matter)

- **Keep the person's hand.** The point is a font that looks like *their* handwriting.
  Prefer adjusting real outlines (`lighten`, `cap_width`, picking a better sample) over
  anything that redraws letters. `build --monoline` gives an even stroke but loses the
  hand-written look; people have rejected it. Only use it if asked.
- **Their handwriting is private.** Everything runs locally. Don't send their photos or
  glyph images to any web service. Notebook pages can contain diary text: use them for
  letter shapes, and don't transcribe or quote their content back.
- **One project folder per person**, outside this repo (e.g. `~/my-font`). Never commit
  project data (`data/`, `out/`, `settings.json`) into this repository. If they want it
  versioned, suggest a separate *private* repo.
- **Show, then tell.** After each change, render it, look at it, and show a before/after.
  Report problems you see, including ones you didn't fix.

## 1. Set up (once)

```bash
test -x .venv/bin/python || { python3 -m venv .venv && .venv/bin/pip install -r requirements.txt; }
command -v tesseract magick || echo "missing: brew install tesseract imagemagick"
```
Ask the user to run installs that need their password. Pick the project folder with them.
It's created on first use, with `settings.json` from `presets/default.json` (font name
"My Handwriting"). Rename the font there.

## 2. Get handwriting in

**Best: a sample sheet.** Gives a complete font from one photo. Ask them to copy the rows
of `layouts/charsheet.txt` onto paper: one character per grid square, dark pen, with clear
gaps (touching characters merge). Writing the whole block twice, e.g. with two pens, gives
a choice per character.

```bash
.venv/bin/python fontkit.py --project ~/my-font ingest-sheet PHOTO.heic [--copies 2] [--rotate cw|ccw|180]
```
Then **Read `<project>/out/review/sheet_<photo-name>.png`** and check every box carries
the right label. The command lists skipped rows (glyph count ≠ layout). Typical causes:
touching or broken characters, stray marks, sideways photo (`--rotate`), or the user
wrote a different row order (copy the layout, edit it, pass `--layout`).

**Also possible: ordinary notebook pages** (`ingest DIR`). OCR-based, works for neat
printed writing, poor for capitals, digits and cursive. Read
`references/notebook-pages.md` before using it.

## 3. Build and look

```bash
.venv/bin/python fontkit.py --project ~/my-font build          # out/<Name>.ttf + .woff2 + specimen.png
.venv/bin/python fontkit.py --project ~/my-font build --otf    # .otf too
```
Read `<project>/out/specimen.png`. Look at it like a reader: does each letter look like
their writing and like *that letter*? Then measure:

```bash
.venv/bin/python .claude/skills/make-handwriting-font/scripts/font_report.py <project>/out/<Name>.ttf
```
It reports stroke-weight variation, capitals vs lowercase weight, slant spread and
outlier glyphs. To see any text or compare versions side by side:
```bash
.venv/bin/python .claude/skills/make-handwriting-font/scripts/render_text.py OUT.png --font A.ttf [--font B.ttf] ["text" ...]
```
Render at a large size (150+) when inspecting individual letters. Small sizes hide defects.

## 4. Fix what's wrong

Common problems and the right lever. Details and recipes are in `references/fixing-glyphs.md`:

| You see | Do |
|---|---|
| a letter is the wrong character, ragged, blobby or from a faint pen | `sheet <chars>`, Read `out/review/sheet.png`, then `pick <char> <id>` or `reject <id>` |
| capitals look wide or heavy next to lowercase | `cap_width` (e.g. 0.85 → 0.8), `cap_lighten` in `settings.json` |
| whole font too bold or too light | `lighten` in `settings.json` (higher = lighter) |
| inconsistent weight/slant across letters | `build --preset neat`: separate "‹Name› Neat" font in `out/neat/` |
| missing characters (`build` prints `missing:`) | add them to the sheet layout and re-ingest, or `region`/`label` |

Rebuild after each change, then look again. Picks always win over automatic choices.
`pick` refuses IDs that don't exist, so copy IDs from the contact sheet.

## 5. Deliver

- Tell them where the files are and show the final specimen.
- **Installing** copies into `~/Library/Fonts` (macOS). Do it only when asked. Install one
  format (the `.ttf`); having both `.ttf` and `.otf` of the same name makes Font Book warn
  about duplicates. Apps need a restart. After a rebuild the installed copy is stale until
  reinstalled; say so.
- **Previewing in a browser:** browsers cache fonts and prefer an installed font with the
  same family name. In any HTML preview, load the file with `new FontFace('SomeUniqueName',
  'url(Font.ttf?v=' + Date.now() + ')')` rather than a plain `@font-face` with the real name.
- Validation: `.venv/bin/python -m ots <font> /dev/null` (pip `opentype-sanitizer`) is
  what browsers use. Every build should pass it.
