# Using ordinary notebook pages

Pages of normal writing can add samples, especially many lowercase letters. They are a
supplement to a sample sheet, not a replacement.

## What works and what doesn't

`ingest` cleans each page, runs Tesseract OCR, and uses a word only when its number of
separate ink blobs equals the number of letters OCR read. Consequences:

- **Neat printed writing:** good lowercase coverage. With many samples per letter, the
  median-shape ranking hides occasional mislabels.
- **Capitals and digits:** OCR on handwriting misreads them a lot, so expect few usable or
  wrong ones. Use hand labelling (below).
- **Cursive / joined letters:** words become one blob, the count never matches, almost
  nothing is extracted. Tell the user a sample sheet is the way.
- **Rare letters** (q, z, x, j, 8…): too few samples to outvote a mislabel. Check them.
- **Ink colour:** red/pink/orange ink and yellow highlighter are removed on purpose (they
  are usually annotations). Writing in those colours vanishes. Very light pencil may too.

```bash
.venv/bin/python fontkit.py --project P ingest ~/photos/notebook/     # or individual files; HEIC ok
.venv/bin/python fontkit.py --project P sheet abcdefghijklmnopqrstuvwxyz --out lower.png
```
Read the contact sheet and `reject` wrong samples.

## Privacy

These pages may be a diary. Use them for letter shapes only: don't transcribe, summarize
or quote their text, and only view as much of a page as you need to find letters.
`P/data/pages/` holds the cleaned pages. They stay in the project folder, which must not
be committed to this public repo.

## Hand-labelling letters from a page region

Good sources: headings in capitals, dates for digits.

1. Find the region. Pages are stored full-resolution at `P/data/pages/<page>.png`. Make a
   downscaled copy to look at (`magick in.png -resize 1400x out.png`), find the area, and
   multiply coordinates by the scale factor (full width ÷ 1400).
2. Cut it into numbered pieces:
   ```bash
   .venv/bin/python fontkit.py --project P region <page> X0 Y0 X1 Y1 --tag heading1 [--rot cw|ccw]
   ```
   Read `P/out/review/region_heading1.png`: each piece has a red box and a number.
   Keep the box tight to one line of text; boxes spanning several lines or underlines
   merge pieces. If an underline touches the letters, shrink the box to just above it.
3. Label the pieces, quoting anything with shell characters:
   ```bash
   .venv/bin/python fontkit.py --project P label heading1:0=R heading1:4=Y 'date1:6=3' 'q:18=?'
   ```
   Hand labels outrank OCR samples. Re-running `region` with a different box renumbers
   the pieces, so check the preview again before relabelling.
