#!/usr/bin/env python3
"""Handwriting font builder.

Workflow (all state lives in ./data so it can be improved incrementally):

  ingest <image-dir|image>...   binarize pages + OCR + auto-extract labeled glyphs
  region <page> x0 y0 x1 y1 [--rot cw|ccw] --tag T
                                cut a page region into numbered components and
                                write a preview (out/review/region_T.png) so you
                                can label them by eye
  label T:k=C [T:k=C ...]       label a region component as character C
  sheet [CHARS]                 contact sheet of samples per char (out/review/)
  pick C ID                     force sample ID to be used for C
  reject ID [ID ...]            never use these samples
  build [--name NAME]           build out/<NAME>.ttf/.otf + specimen PNG

Sample IDs are shown under each thumbnail in sheets.
"""
import argparse, collections, csv, glob, json, os, shutil, subprocess, sys
import cv2, numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data')
PAGES = os.path.join(DATA, 'pages')          # binarized page PNGs (black ink on white)
OCR = os.path.join(DATA, 'ocr')              # tesseract TSVs
LIB = os.path.join(DATA, 'library')          # glyph samples: library/<codepoint>/<id>.png
REG = os.path.join(DATA, 'regions')          # manual region crops
SOURCES = os.path.join(DATA, 'sources.json')       # page name -> original image path
REGIONS = os.path.join(DATA, 'regions.json')       # tag -> region definition
LABELS = os.path.join(DATA, 'manual_labels.json')  # "tag:k" -> char
SEL = os.path.join(DATA, 'selections.json')        # {"pick": {char: id}, "reject": [ids]}
INDEX = os.path.join(DATA, 'index.json')           # all samples + metadata
SHEETS = os.path.join(DATA, 'sheets.json')         # character sample sheets: page -> layout/copies
OUT = os.path.join(ROOT, 'out')
PRESETS = os.path.join(ROOT, 'presets')          # default.json + optional variants (e.g. neat.json)
SETTINGS = os.path.join(ROOT, 'settings.json')   # this project's settings (font name, weights)
REVIEW = os.path.join(OUT, 'review')
PROJ = ROOT       # project folder: holds data/, out/ and settings.json

def set_project(path):
    """Point every data/output path at a separate project folder, so each person's
    handwriting is kept apart from the example data that ships with the repo."""
    global PROJ, DATA, PAGES, OCR, LIB, REG, SOURCES, REGIONS, LABELS, SEL, INDEX, SHEETS, OUT, SETTINGS, REVIEW
    PROJ = os.path.abspath(os.path.expanduser(path))
    DATA = os.path.join(PROJ, 'data')
    PAGES, OCR, LIB, REG = (os.path.join(DATA, d) for d in ('pages', 'ocr', 'library', 'regions'))
    SOURCES, REGIONS, LABELS, SEL, INDEX, SHEETS = (os.path.join(DATA, f) for f in (
        'sources.json', 'regions.json', 'manual_labels.json', 'selections.json', 'index.json', 'sheets.json'))
    OUT = os.path.join(PROJ, 'out'); REVIEW = os.path.join(OUT, 'review')
    SETTINGS = os.path.join(PROJ, 'settings.json')
    if not os.path.exists(SETTINGS):   # new project: start from the default preset
        os.makedirs(PROJ, exist_ok=True)
        jsave(SETTINGS, jload(os.path.join(PRESETS, 'default.json'), {}))
        print(f'new project at {PROJ} (edit {SETTINGS} to name your font)')

def jload(p, d):
    return json.load(open(p)) if os.path.exists(p) else d
def jsave(p, o):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(o, open(p, 'w'), indent=1, ensure_ascii=False, default=lambda v: v.item())

def safe(c):  # directory/file-safe name for a char
    return f'U+{ord(c):04X}'

# ---------------------------------------------------------------- binarize
def read_image(path):
    """Read any photo, including iPhone HEIC (converted with metadata stripped)."""
    if path.lower().endswith(('.heic', '.heif')):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = os.path.join(td, 'conv.jpg')
            subprocess.run(['magick', path, '-auto-orient', '-strip', tmp], check=True)
            return cv2.imread(tmp)
    return cv2.imread(path)

def binarize(path_or_img):
    img = read_image(path_or_img) if isinstance(path_or_img, str) else path_or_img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lum = img.max(axis=2).astype(np.float32)   # highlighter/red stay bright in some channel
    bg = cv2.GaussianBlur(cv2.dilate(lum, np.ones((31, 31), np.uint8)), (0, 0), 25)
    norm = np.clip(lum / np.maximum(bg, 1), 0, 1)
    ink = (norm < 0.55).astype(np.uint8) * 255
    red = ((hsv[..., 0] < 12) | (hsv[..., 0] > 165)) & (hsv[..., 1] > 90)
    ink[red] = 0                                # drop red underlines
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return 255 - ink

def merge_components(ink, min_area=25, vgap=0.6):
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    comps = [[st[i][0], st[i][1], st[i][0] + st[i][2], st[i][1] + st[i][3], [i]]
             for i in range(1, n) if st[i][4] >= min_area]
    comps.sort(key=lambda c: c[0]); merged = []
    for c in comps:
        if merged:
            m = merged[-1]
            ov = min(m[2], c[2]) - max(m[0], c[0])
            vg = max(m[1], c[1]) - min(m[3], c[3])
            if ov > 0.5 * min(m[2] - m[0], c[2] - c[0]) and vg < vgap * max(m[3] - m[1], c[3] - c[1]):
                merged[-1] = [min(m[0], c[0]), min(m[1], c[1]), max(m[2], c[2]), max(m[3], c[3]), m[4] + c[4]]
                continue
        merged.append(c)
    return lab, merged

def comp_mask(lab, m):
    return np.isin(lab[m[1]:m[3], m[0]:m[2]], m[4]).astype(np.uint8) * 255

# ---------------------------------------------------------------- ingest
def cmd_ingest(a):
    files = []
    for p in a.paths:
        if os.path.isdir(p):
            files += sorted(f for f in glob.glob(os.path.join(p, '*'))
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.heic', '.heif')))
        else:
            files.append(p)
    src = jload(SOURCES, {})
    os.makedirs(PAGES, exist_ok=True); os.makedirs(OCR, exist_ok=True)
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        src[name] = os.path.abspath(f)
        pb = os.path.join(PAGES, name + '.png')
        if not os.path.exists(pb) or a.force:
            cv2.imwrite(pb, binarize(f)); print('binarized', name)
        tsv = os.path.join(OCR, name)
        if not os.path.exists(tsv + '.tsv') or a.force:
            subprocess.run(['tesseract', pb, tsv, '--psm', '3', 'tsv'], capture_output=True)
            print('ocr', name)
    jsave(SOURCES, src)
    reindex()

NODESC = set('abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
XH = set('acemnorsuvwxz')

def auto_extract(name):
    """OCR words whose blob count == letter count -> one labeled sample per blob."""
    img = cv2.imread(os.path.join(PAGES, name + '.png'), 0)
    ink = (img < 128).astype(np.uint8); H, W = ink.shape
    rows = list(csv.reader(open(os.path.join(OCR, name + '.tsv')), delimiter='\t', quoting=csv.QUOTE_NONE))
    out = []
    for r in rows[1:]:
        if len(r) < 12 or r[0] != '5' or float(r[10]) < 50: continue
        txt = r[11].strip().strip('"\'.,:;!?()[]“”‘’-—|')
        if not txt or not txt.isalnum() or not txt.isascii() or len(txt) > 14: continue
        x, y, w, h = map(int, r[6:10]); p = 6
        x0, y0, x1, y1 = max(0, x - p), max(0, y - p), min(W, x + w + p), min(H, y + h + p)
        lab, merged = merge_components(ink[y0:y1, x0:x1])
        big = [m for m in merged if (m[3] - m[1]) > 0.15 * h or (m[2] - m[0]) > 0.3 * h]
        if len(big) != len(txt): continue
        bots = [m[3] for m, c in zip(big, txt) if c in NODESC]
        tops = [m[1] for m, c in zip(big, txt) if c in XH]
        if not bots: continue
        base = float(np.median(bots)); xh = base - float(np.median(tops)) if tops else None
        for m, c in zip(big, txt):
            sid = f'{name}@{x0 + m[0]},{y0 + m[1]}'
            out.append(dict(id=sid, char=c, src='auto', conf=float(r[10]),
                            top=(m[1] - base) / xh if xh else None,
                            bottom=(m[3] - base) / xh if xh else None,
                            mask=comp_mask(lab, m)))
    return out

def region_components(tag):
    d = jload(REGIONS, {})[tag]
    ink = (cv2.imread(os.path.join(PAGES, d['page'] + '.png'), 0) < 128).astype(np.uint8)
    x0, y0, x1, y1 = d['box']; ink = ink[y0:y1, x0:x1]
    if d.get('rot') == 'cw': ink = cv2.rotate(ink, cv2.ROTATE_90_CLOCKWISE)
    if d.get('rot') == 'ccw': ink = cv2.rotate(ink, cv2.ROTATE_90_COUNTERCLOCKWISE)
    lab, merged = merge_components(ink, min_area=40)
    return ink, lab, merged

def reindex():
    """Rebuild the sample library from pages + OCR + manual labels.

    Samples whose source page image is not available locally (e.g. a fresh clone, where
    private source photos are git-ignored) are carried over unchanged from the
    existing library instead of being dropped.
    """
    have_page = lambda name: os.path.exists(os.path.join(PAGES, name + '.png'))
    regs = jload(REGIONS, {})
    def source_page(entry):
        if entry['src'] == 'manual':
            return regs.get(entry['id'][2:].rsplit(':', 1)[0], {}).get('page', '')
        if entry['src'] == 'sheet':
            return entry['id'].split(':')[1]
        return entry['id'].split('@')[0]
    # keep samples from pages we can't regenerate (read their images before wiping)
    kept = []
    for e in jload(INDEX, []):
        f = os.path.join(DATA, e.get('file', ''))
        if not have_page(source_page(e)) and os.path.exists(f):
            e = dict(e); e['mask'] = 255 - cv2.imread(f, 0); kept.append(e)
    if os.path.exists(LIB): shutil.rmtree(LIB)
    idx = []
    for tsv in sorted(glob.glob(os.path.join(OCR, '*.tsv'))):
        name = os.path.basename(tsv)[:-4]
        if have_page(name): idx += auto_extract(name)
    for name in jload(SHEETS, {}):
        if have_page(name): idx += sheet_glyphs(name)[0]
    labels = jload(LABELS, {})
    cache = {}
    for key, c in labels.items():
        tag, k = key.rsplit(':', 1)
        if tag not in regs or not have_page(regs[tag]['page']): continue
        if tag not in cache: cache[tag] = region_components(tag)
        _, lab, merged = cache[tag]
        idx.append(dict(id=f'm:{key}', char=c, src='manual', conf=100, top=None, bottom=None,
                        mask=comp_mask(lab, merged[int(k)])))
    fresh = {e['id'] for e in idx}
    for e in kept:   # a re-labelled manual sample replaces its carried-over copy
        if e['id'] not in fresh and (e['src'] != 'manual' or e['id'][2:] in labels):
            if e['src'] == 'manual': e['char'] = labels[e['id'][2:]]
            idx.append(e)
    for s in idx:
        d = os.path.join(LIB, safe(s['char'])); os.makedirs(d, exist_ok=True)
        s['file'] = os.path.relpath(os.path.join(d, s['id'].replace(':', '_').replace('@', '_').replace(',', '_') + '.png'), DATA)
        cv2.imwrite(os.path.join(DATA, s['file']), 255 - s.pop('mask'))
    jsave(INDEX, idx)
    cnt = collections.Counter(s['char'] for s in idx)
    print(f'{len(idx)} samples, {len(cnt)} chars:', ''.join(sorted(cnt)))

# ---------------------------------------------------------------- character sample sheets
BASESIT = set('ABCDEFGHIJKLMNOPRSTUVWXYZabcdefhiklmnorstuvwxz0123456789.!?:&#%@$₹|/\\')
XHSET = set('acemnorsuvwxz')

def read_layout(path, copies):
    rows = []
    for line in open(path, encoding='utf-8'):
        if line.strip() and not line.lstrip().startswith('#'):
            rows.append(line.split())
    return rows * copies

def sheet_glyphs(name):
    """Segment a sample-sheet page into per-cell glyphs and label them from its layout.

    Returns (samples, report, boxes). Rows whose glyph count doesn't match the layout
    are skipped and reported rather than guessed.
    """
    spec = jload(SHEETS, {})[name]
    layout = read_layout(os.path.join(ROOT, spec['layout']), spec.get('copies', 1))
    ink = (cv2.imread(os.path.join(PAGES, name + '.png'), 0) < 128).astype(np.uint8)
    H, W = ink.shape
    n, lab, st, cen = cv2.connectedComponentsWithStats(ink, 8)
    big = [st[i][3] for i in range(1, n) if st[i][4] > 300]
    Hm = float(np.median(big)) if big else 50.0          # typical glyph height (px)
    Am = float(np.median([st[i][4] for i in range(1, n) if st[i][4] > 300] or [1000]))
    dt = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    thick = np.zeros(n); np.maximum.at(thick, lab.ravel(), dt.ravel())  # half stroke width per comp
    r_med = float(np.median([thick[i] for i in range(1, n) if st[i][4] > 300] or [3.0]))
    mx, my = int(0.03 * W), int(0.02 * H)
    keep = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if thick[i] < 0.45 * r_med: continue                         # hairline grid dashes / paper texture
        if x < mx or y < my or x + w > W - mx or y + h > H - my: continue  # page margins / shadows
        if a < max(30, 0.02 * Am): continue                          # specks
        if x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2: continue  # page edges / shadows
        if max(w, h) > 3 * Hm and min(w, h) < 0.15 * max(w, h): continue  # ruled / grid lines
        if a > 25 * Am: continue                                     # big blobs
        keep.append(i)
    # union parts of one character: close horizontally AND vertically
    parent = {i: i for i in keep}
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for ai, i in enumerate(keep):
        xi, yi, wi, hi, _ = st[i]
        for j in keep[ai + 1:]:
            xj, yj, wj, hj, _ = st[j]
            gx = max(xi, xj) - min(xi + wi, xj + wj)
            gy = max(yi, yj) - min(yi + hi, yj + hj)
            if gx < 0.35 * Hm and gy < 0.6 * Hm: parent[find(i)] = find(j)
    groups = collections.defaultdict(list)
    for i in keep: groups[find(i)].append(i)
    boxes = []
    for ids in groups.values():
        x0 = min(st[i][0] for i in ids); y0 = min(st[i][1] for i in ids)
        x1 = max(st[i][0] + st[i][2] for i in ids); y1 = max(st[i][1] + st[i][3] for i in ids)
        boxes.append([x0, y0, x1, y1, ids])
    # rows: cluster by vertical centre
    boxes.sort(key=lambda b: (b[1] + b[3]) / 2)
    rows = []
    for b in boxes:
        cy = (b[1] + b[3]) / 2
        if rows and cy - rows[-1][-1][5] < 0.75 * Hm: rows[-1].append(b + [cy])
        else: rows.append([b + [cy]])
    for r in rows: r.sort(key=lambda b: b[0])
    report, out, labelled = [], [], []
    if len(rows) != len(layout):
        report.append(f'found {len(rows)} rows, layout has {len(layout)}: rows matched by order until mismatch')
    # x-height from lowercase rows that match
    xhs = [b[3] - b[1] for r, chars in zip(rows, layout) if len(r) == len(chars)
           for b, c in zip(r, chars) if c in XHSET]
    xh = float(np.median(xhs)) if xhs else 0.6 * Hm
    # one page-wide baseline slope (from rows with many base-sitting glyphs), per-row offset
    slopes = []
    for r, chars in zip(rows, layout):
        pts = [((b[0] + b[2]) / 2, b[3]) for b, c in zip(r, chars) if c in BASESIT] if len(r) == len(chars) else []
        if len(pts) >= 8: slopes.append(np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)[0])
    k = float(np.median(slopes)) if slopes else 0.0
    for ri, (r, chars) in enumerate(zip(rows, layout)):
        if len(r) != len(chars):
            report.append(f'row {ri + 1}: found {len(r)} glyphs, layout expects {len(chars)} ({" ".join(chars)}) - skipped')
            continue
        pts = [((b[0] + b[2]) / 2, b[3]) for b, c in zip(r, chars) if c in BASESIT] or [((b[0] + b[2]) / 2, b[3]) for b in r]
        c0 = float(np.median([y - k * x for x, y in pts]))
        for ci, (b, c) in enumerate(zip(r, chars)):
            base = k * (b[0] + b[2]) / 2 + c0
            m = np.isin(lab[b[1]:b[3], b[0]:b[2]], b[4]).astype(np.uint8) * 255
            out.append(dict(id=f's:{name}:{ri + 1}:{ci + 1}', char=c, src='sheet', conf=100,
                            top=(b[1] - base) / xh, bottom=(b[3] - base) / xh, mask=m))
            labelled.append((b, c))
    return out, report, labelled

def cmd_ingest_sheet(a):
    name = a.name or os.path.splitext(os.path.basename(a.image))[0]
    img = read_image(a.image)
    rot = {'cw': cv2.ROTATE_90_CLOCKWISE, 'ccw': cv2.ROTATE_90_COUNTERCLOCKWISE, '180': cv2.ROTATE_180}
    if a.rotate: img = cv2.rotate(img, rot[a.rotate])
    os.makedirs(PAGES, exist_ok=True)
    cv2.imwrite(os.path.join(PAGES, name + '.png'), binarize(img))
    src = jload(SOURCES, {}); src[name] = os.path.abspath(a.image); jsave(SOURCES, src)
    sheets = jload(SHEETS, {})
    sheets[name] = dict(layout=os.path.relpath(os.path.abspath(a.layout), ROOT), copies=a.copies, rotate=a.rotate)
    jsave(SHEETS, sheets)
    samples, report, labelled = sheet_glyphs(name)
    vis = cv2.cvtColor(cv2.imread(os.path.join(PAGES, name + '.png'), 0), cv2.COLOR_GRAY2BGR)
    for b, c in labelled:
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        t = c if c.isascii() else f'U+{ord(c):04X}'
        cv2.putText(vis, t, (b[0], max(30, b[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 0, 0), 3)
    os.makedirs(REVIEW, exist_ok=True); p = os.path.join(REVIEW, f'sheet_{name}.png'); cv2.imwrite(p, vis)
    print(f'{len(samples)} glyphs labelled -> check {p}')
    for line in report: print('  !', line)
    reindex()

# ---------------------------------------------------------------- manual regions
def cmd_region(a):
    regs = jload(REGIONS, {})
    regs[a.tag] = dict(page=a.page, box=[a.x0, a.y0, a.x1, a.y1], rot=a.rot)
    jsave(REGIONS, regs)
    ink, lab, merged = region_components(a.tag)
    vis = cv2.cvtColor((1 - ink) * 255, cv2.COLOR_GRAY2BGR)
    for k, m in enumerate(merged):
        cv2.rectangle(vis, (m[0], m[1]), (m[2], m[3]), (0, 0, 255), 2)
        cv2.putText(vis, str(k), (m[0], max(20, m[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    os.makedirs(REVIEW, exist_ok=True)
    p = os.path.join(REVIEW, f'region_{a.tag}.png'); cv2.imwrite(p, vis)
    print(f'{len(merged)} components -> {p}\nlabel with: fontkit.py label {a.tag}:<k>=<char>')

def cmd_label(a):
    labels = jload(LABELS, {})
    for item in a.items:
        key, c = item.rsplit('=', 1)
        if len(c) != 1: sys.exit(f'bad label {item}')
        labels[key] = c
    jsave(LABELS, labels); reindex()

# ---------------------------------------------------------------- selection
def norm_vec(path):
    g = 255 - cv2.imread(path, 0); h, w = g.shape; s = max(h, w)
    c = np.zeros((s, s), np.uint8); c[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = g
    return cv2.GaussianBlur(cv2.resize(c, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255, (0, 0), 1.5).ravel()

def sheet_copy(sample):
    """Which repetition of the layout (e.g. which pen) a sheet sample comes from."""
    _, name, row, _ = sample['id'].split(':')
    spec = jload(SHEETS, {}).get(name, {})
    n = len(read_layout(os.path.join(ROOT, spec['layout']), 1)) if spec else 1
    return (int(row) - 1) // n + 1

def cleanest_copy():
    """The sheet repetition (pen) whose glyphs have the fewest specks/holes/ragged edges."""
    per = collections.defaultdict(list)
    for x in jload(INDEX, []):
        if x['src'] == 'sheet': per[sheet_copy(x)].append(ink_defects(x))
    return min(per, key=lambda k: np.mean(per[k])) if per else None

def ink_defects(sample):
    """Speckle fragments + pin-holes + edge raggedness: high for faint / dry pens."""
    m = (cv2.imread(os.path.join(DATA, sample['file']), 0) < 128).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    frags = sum(1 for i in range(1, n) if st[i][4] < 0.04 * m.sum())
    n2, _, s2, _ = cv2.connectedComponentsWithStats(1 - m, 4)
    holes = sum(1 for i in range(1, n2)
                if s2[i][0] > 0 and s2[i][1] > 0 and s2[i][0] + s2[i][2] < m.shape[1]
                and s2[i][1] + s2[i][3] < m.shape[0] and s2[i][4] < 0.02 * m.size)
    # ragged / bitten edges (dry pen): outline length vs. a lightly smoothed outline
    mp = cv2.copyMakeBorder(m, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
    per = lambda a: sum(cv2.arcLength(c, True) for c in
                        cv2.findContours(a, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0])
    smooth = (cv2.GaussianBlur(mp.astype(np.float32), (0, 0), 2.0) > 0.5).astype(np.uint8)
    rough = per(mp) / max(per(smooth), 1)
    return frags + holes + int(round(max(0.0, rough - 1.0) / 0.05))

def ranked(prefer_copy=None):
    """char -> samples best-first: your pick, then sample-sheet glyphs, then manual labels,
    then OCR-extracted ones. Within sheet and OCR samples, the one closest to the median
    shape of *all* samples of that char (i.e. your everyday writing) comes first."""
    idx = jload(INDEX, []); sel = jload(SEL, {'pick': {}, 'reject': []})
    rej = set(sel['reject']); by = collections.defaultdict(list)
    for s in idx:
        if s['id'] not in rej: by[s['char']].append(s)
    res = {}
    for c, ss in by.items():
        if len(ss) >= 3:
            V = np.array([norm_vec(os.path.join(DATA, s['file'])) for s in ss]); med = np.median(V, 0)
            dist = dict(zip((s['id'] for s in ss), ((V - med) ** 2).sum(1)))
        else:
            dist = {s['id']: 0.0 for s in ss}
        sheet = [x for x in ss if x['src'] == 'sheet']
        if len(sheet) > 1:   # between pens/copies, only the cleanest strokes compete
            dfx = {x['id']: ink_defects(x) for x in sheet}; best = min(dfx.values())
            ss = [x for x in ss if x['src'] != 'sheet' or dfx[x['id']] <= best
                  or x['id'] == sel['pick'].get(c)]          # an explicit pick always survives
        tier = {'sheet': 0, 'manual': 1, 'auto': 2}
        pick = sel['pick'].get(c)
        def copy_miss(x):
            if not prefer_copy or x['src'] != 'sheet': return 0
            return int(sheet_copy(x) != prefer_copy)
        res[c] = sorted(ss, key=lambda s: (s['id'] != pick, tier[s['src']], copy_miss(s), dist[s['id']]))
    return res

def cmd_pick(a):
    ids = {x['id']: x['char'] for x in jload(INDEX, [])}
    if ids.get(a.id) != a.char:
        sys.exit(f'no sample {a.id!r} for {a.char!r} (see `fontkit.py sheet {a.char}`)')
    sel = jload(SEL, {'pick': {}, 'reject': []}); sel['pick'][a.char] = a.id; jsave(SEL, sel)
def cmd_reject(a):
    sel = jload(SEL, {'pick': {}, 'reject': []}); sel['reject'] = sorted(set(sel['reject']) | set(a.ids)); jsave(SEL, sel)

def cmd_sheet(a):
    R = ranked(); chars = a.chars or ''.join(sorted(R)); N, cell = 10, 120
    rows = []
    for c in chars:
        row = np.full((cell + 30, cell * (N + 1)), 255, np.uint8)
        cv2.putText(row, c, (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 3)
        for k, s in enumerate(R.get(c, [])[:N]):
            g = cv2.imread(os.path.join(DATA, s['file']), 0); h, w = g.shape; sc = (cell - 10) / max(h, w)
            g = cv2.resize(g, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA)
            x = cell * (k + 1) + 5; row[5:5 + g.shape[0], x:x + g.shape[1]] = g
            short = s['id'] if s['src'] == 'manual' else s['id'].split('_')[-1]
            cv2.putText(row, short[-16:], (cell * (k + 1) + 2, cell + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 90, 1)
        cv2.line(row, (0, cell + 29), (row.shape[1], cell + 29), 200, 1); rows.append(row)
    os.makedirs(REVIEW, exist_ok=True)
    p = os.path.join(REVIEW, a.out or 'sheet.png'); cv2.imwrite(p, np.vstack(rows)); print(p)

# ---------------------------------------------------------------- font build
UPM, XHT, CAP, ASC, DESC = 1000, 420, 640, 690, -250
ASCENDERS = set('bdfhklt') | set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789?!&/₹')
DESCENDERS = set('gjpqy')

# Where symbols sit. Hand-written symbol rows rarely share a clean baseline, so the
# vertical position is typographic; small marks keep their written height.
#   ('span', bottom, top)   scale glyph to exactly this span
#   ('center', y)           keep written height, centred at y
#   ('base', y) / ('top', y) keep written height, bottom / top at y
SYMBOL_POS = {
    **{c: ('span', -130, 700) for c in '()[]{}'},
    **{c: ('span', -100, 700) for c in '/\\|'},
    **{c: ('span', 0, CAP) for c in '!?&#%'},
    '@': ('span', -90, 560), '$': ('span', -70, 710),
    '.': ('base', 0), ',': ('top', 95),
    '-': ('center', 230), '+': ('center', 250), '=': ('center', 240), '~': ('center', 260),
    '*': ('top', CAP), '^': ('top', CAP), "'": ('top', CAP), '"': ('top', CAP), '`': ('top', CAP),
}

def symbol_span(c, s):
    kind, *v = SYMBOL_POS[c]
    if kind == 'span': return v[0], v[1]
    h = (s['bottom'] - s['top']) * XHT if s.get('top') is not None else 120
    h = max(h, 40)
    if kind == 'center': return v[0] - h / 2, v[0] + h / 2
    if kind == 'base': return v[0], v[0] + h
    return v[0] - h, v[0]                       # 'top'

def vmetrics(c):
    """(bottom, top) in font units for a character class."""
    if c == 't': return 0, 560
    if c in 'Q': return -60, CAP
    if c in 'J': return -40, CAP
    if c in ASCENDERS: return 0, CAP if not c.islower() else ASC
    if c in DESCENDERS: return DESC, XHT
    if c in "'\"": return CAP - 180, CAP
    if c == '/': return -40, CAP + 20
    return 0, XHT

def smooth_mask(mask, target_h):
    """Upsample + blur + threshold -> smooth anti-jagged outline at ~target_h px."""
    h, w = mask.shape; sc = target_h / h
    m = cv2.copyMakeBorder(mask, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
    big = cv2.resize(m.astype(np.float32), None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
    sig = max(0.8, 0.55 * sc)   # light: smooths pixel steps without closing 2-3px gaps
    big = cv2.GaussianBlur(big, (0, 0), sig)
    out = (big > 127).astype(np.uint8)
    # fill pin-holes left by uneven ink, keep real counters (o, e, a ...)
    cnts, hier = cv2.findContours(out, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    for i, cn in enumerate(cnts):
        if hier[0][i][3] >= 0 and cv2.contourArea(cn) < (0.004 * out.size):
            cv2.drawContours(out, [cn], -1, 1, -1)
    return out, 4 * sc

def mask_to_contours(mask, units_per_px, y_off, smooth=0.012):
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    res = []
    if hier is None: return res
    h = mask.shape[0]
    for i, cn in enumerate(cnts):
        if cv2.contourArea(cn) < 30: continue
        # smooth along the outline (circular Gaussian): removes pixel-step wobble without
        # ever merging neighbouring strokes the way image blurring can
        pts_px = cn.reshape(-1, 2).astype(np.float32)
        sg = max(1.0, smooth * h)
        if len(pts_px) > 6 * sg:
            k = np.exp(-0.5 * (np.arange(-int(3 * sg), int(3 * sg) + 1) / sg) ** 2); k /= k.sum()
            pad_n = len(k) // 2
            ext = np.vstack([pts_px[-pad_n:], pts_px, pts_px[:pad_n]])
            pts_px = np.stack([np.convolve(ext[:, d], k, mode="valid") for d in (0, 1)], 1).astype(np.float32)
        ap = cv2.approxPolyDP(pts_px.reshape(-1, 1, 2), 0.6, True).reshape(-1, 2)
        pts = [(float(x) * units_per_px, float(h - y) * units_per_px + y_off) for x, y in ap]
        area = sum(pts[j][0] * pts[j - 1][1] - pts[j - 1][0] * pts[j][1] for j in range(len(pts))) / 2
        hole = hier[0][i][3] >= 0
        # TrueType: outer contours clockwise (negative area), holes counter-clockwise
        if (area > 0) != hole: pts = pts[::-1]
        res.append(pts)
    return res

def load_mask(s):
    m = (cv2.imread(os.path.join(DATA, s['file']), 0) < 128).astype(np.uint8)
    # drop specks/fragments (< 4% of the ink) that would distort the glyph's size;
    # real secondary parts such as the dot on '?' are bigger than that
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = [i for i in range(1, n) if st[i][4] >= 0.04 * m.sum()]
    m = np.isin(lab, keep).astype(np.uint8)
    ys, xs = np.nonzero(m)
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return fill_dry_ink(m, s['char']) * 255

# Characters whose shape legitimately has enclosed holes (counters).
COUNTERS = set('abdegopqABDOPQR04689&@%#?')
SINGLE_STROKE = set('Il1|!ij')
STROKE_PX = 7.6   # typical pen stroke width in source pixels (median over lowercase samples)

def fill_holes(m, max_r):
    """Fill enclosed holes narrower than max_r x stroke width (scale-free: uses the
    glyph's own stroke width)."""
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    ridge = dt[(dt >= cv2.dilate(dt, np.ones((3, 3), np.float32))) & (m > 0)]
    sw = 2 * float(np.median(ridge)) if ridge.size else 1.0
    inv = (1 - m).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, 4)
    out = m.copy()
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        if x == 0 or y == 0 or x + w >= m.shape[1] or y + h >= m.shape[0]: continue
        hole = (lab == i).astype(np.uint8)
        if cv2.distanceTransform(hole, cv2.DIST_L2, 3).max() < max_r * sw: out[hole > 0] = 1
    return out

def fill_dry_ink(m, c):
    """Repair 'running out of ink' artefacts: cracks, pin-holes, hollow outlined strokes.

    Gaps left by dry ink are thin (inscribed radius << stroke width); real counters
    (o, e, a, 6 ...) are wide (>= 0.3 x stroke). Characters with no counters get every
    enclosed hole filled; characters with counters only lose the thin ones.
    """
    pad = 6
    m = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)            # bridge hairline cracks
    if c in SINGLE_STROKE:                                 # one stroke, nothing nearby to merge:
        ks = max(3, int(round(0.9 * STROKE_PX)))           # close bites and gaps firmly
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
    inv = (1 - m).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, 4)
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        if x == 0 or y == 0 or x + w >= m.shape[1] or y + h >= m.shape[0]:
            continue                                        # background, not a hole
        hole = (lab == i).astype(np.uint8)
        r = cv2.distanceTransform(hole, cv2.DIST_L2, 3).max() / STROKE_PX
        if c not in COUNTERS or r < 0.22:
            m[hole > 0] = 1
    return m[pad:-pad, pad:-pad]

LIGHTEN = 20   # font units shaved off every stroke's thickness (set by `build --lighten`)
MONOLINE = False  # opt-in: redraw glyphs with one uniform pen (`build --monoline`)
WEIGHT = 100   # monoline pen thickness in font units (`build --weight`)

def prune_spurs(sk, L):
    """Remove skeleton branches shorter than L px (blob corners), keep real stroke ends."""
    from scipy.ndimage import convolve
    K = np.ones((3, 3), np.uint8); K[1, 1] = 0
    ends = lambda a: a & (convolve(a.astype(np.uint8), K, mode='constant') == 1)
    cur = sk.copy()
    for _ in range(L):
        cur &= ~ends(cur)
    grow = ends(cur)
    seed = cur.copy()
    for _ in range(L):                       # regrow only from surviving ends
        grow = cv2.dilate(grow.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & sk & ~seed
        if not grow.any(): break
        seed |= grow
    return seed

CAP_WIDTH = 0.9   # horizontal scale for capitals (set by `build --cap-width`)

def condense(m, xscale):
    """Narrow a glyph horizontally while keeping its stroke weight: squeeze the width,
    then thicken horizontally by the amount vertical strokes lost."""
    if xscale >= 0.999: return m
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    sw = 2 * float(np.percentile(dt[m > 0], 90)) if m.any() else 0
    sq = cv2.resize(m.astype(np.float32), (max(1, int(round(m.shape[1] * xscale))), m.shape[0]),
                    interpolation=cv2.INTER_AREA)
    sq = (sq > 0.5).astype(np.uint8)
    k = int(round(sw * (1 - xscale)))
    if k >= 1:
        sq = cv2.dilate(sq, np.ones((1, k + 1), np.uint8))
    return sq

def build_glyph(mask, bottom, top, xscale=1.0, extra_lighten=0.0, char=None):
    """Natural outline of the handwritten glyph, evenly thinned by LIGHTEN units.

    Thinning peels the same amount off every edge, so the pen's natural thick/thin
    variation is preserved. The glyph is scaled so that after thinning it still spans
    exactly bottom..top.
    """
    if MONOLINE:
        return build_glyph_monoline(mask, bottom, top)
    L = LIGHTEN + extra_lighten
    hu = top - bottom
    m, pad = smooth_mask(mask, 220)
    ink_h = m.shape[0] - 2 * pad
    upp = (hu + L) / ink_h
    if L > 0:
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        ridge = dt[(dt >= cv2.dilate(dt, np.ones((3, 3), np.float32))) & (m > 0)]
        r_stroke = float(np.median(ridge)) if ridge.size else 0.0
        r = min(L / 2 / upp, 0.35 * r_stroke)   # thin, but never cut through a thin stroke
        thin = (dt > r).astype(np.uint8)
        if thin.sum() > 0.3 * m.sum():      # never let thin glyphs disappear
            m = thin
        if char is not None and char not in COUNTERS:   # thinning must not open new holes
            m = fill_holes(m, 1.0 if char in SINGLE_STROKE else 0.22)
    m = condense(m, xscale)
    contours = mask_to_contours(m, upp, bottom - L / 2 - pad * upp)
    return contours, (m.shape[1] - 2 * pad) * upp - L, -pad * upp + L / 2

def build_glyph_monoline(mask, bottom, top):
    """Trace the glyph's centreline and re-draw it with a uniform round pen of WEIGHT units.

    Gives every glyph the same stroke weight regardless of how heavily the ink went
    down or how the photo binarized, and removes leftover ragged / dry-ink edges.
    """
    from skimage.morphology import skeletonize
    hu = top - bottom
    m, pad = smooth_mask(mask, 220)
    sc = 220 / mask.shape[0]
    sk = prune_spurs(skeletonize(m > 0), max(2, int(round(0.7 * STROKE_PX * sc))))
    if not sk.any(): sk = m > 0
    ys, xs = np.nonzero(sk)
    sh, sw = ys.max() - ys.min(), xs.max() - xs.min()
    upp = (hu - WEIGHT) / max(sh, 1)                     # stroke centre spans height - weight
    r = WEIGHT / 2 / upp                                 # pen radius in px
    P = int(np.ceil(r)) + 4
    canvas = np.zeros((sh + 1 + 2 * P, sw + 1 + 2 * P), np.uint8)
    canvas[ys - ys.min() + P, xs - xs.min() + P] = 1
    dt = cv2.distanceTransform((1 - canvas).astype(np.uint8), cv2.DIST_L2, 5)
    out = cv2.GaussianBlur((dt <= r).astype(np.float32), (0, 0), max(1.0, r * 0.25))
    out = (out > 0.5).astype(np.uint8)
    # canvas row (P + sh) is the skeleton's lowest point -> font y = bottom + WEIGHT/2
    contours = mask_to_contours(out, upp, bottom + WEIGHT / 2 - (P + 1) * upp)
    return contours, sw * upp + WEIGHT, P * upp - WEIGHT / 2

# ---------------------------------------------------------------- "neat" variant
NEAT = {}            # filled from settings when `neat: true`
NO_NORM = set(".,:;'\"`-~^*")                     # marks: stroke weight not meaningful
SLANT_SET = set('bdfhijklmnpqrtuyBDEFHIJKLMNPRTU147|!')  # glyphs with a measurable stem
ROUND = set('oceadbpqgsOCGQS0689')

def shear(m, t):
    """x' = x + t*(H-1-y): positive t leans the glyph to the right."""
    H, W = m.shape
    off = max(0.0, -t * (H - 1))
    Wn = int(np.ceil(W + abs(t) * (H - 1))) + 2
    M = np.float32([[1, -t, t * (H - 1) + off], [0, 1, 0]])
    return (cv2.warpAffine(m.astype(np.float32), M, (Wn, H), flags=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)

def estimate_slant(m):
    """Lean (degrees, + = right) that makes vertical strokes most vertical."""
    best, best_a = -1, 0.0
    for a in range(-30, 31):
        proj = shear(m, -np.tan(np.radians(a))).sum(0).astype(np.float64)
        sc = (proj ** 2).sum()
        if sc > best: best, best_a = sc, float(a)
    return best_a

def stroke_px(m):
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    ridge = dt[(dt >= cv2.dilate(dt, np.ones((3, 3), np.float32))) & (m > 0)]
    return 2 * float(np.median(ridge)) if ridge.size else 0.0

def build_glyph_neat(mask, bottom, top, xscale=1.0, char=None, slant_fix=0.0):
    """Natural outline, made consistent: slant nudged toward the common slant, stroke
    weight nudged toward a common target (by thinning/thickening the real outline, so
    each glyph keeps its own thick/thin), then smoothed along the contour."""
    hu = top - bottom
    m, _ = smooth_mask(mask, 220)
    m = cv2.copyMakeBorder(m, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=0)
    if abs(slant_fix) > 0.2:
        m = shear(m, np.tan(np.radians(slant_fix)))
    ys, _ = np.nonzero(m); ink_h0 = ys.max() - ys.min() + 1
    if char not in NO_NORM and NEAT.get('stroke_strength', 0) > 0:
        s_, T, sw = NEAT['stroke_strength'], NEAT['stroke_target'], stroke_px(m)
        # change d (px) so that, after rescaling to height hu, stroke -> s*T + (1-s)*own
        d = s_ * (T * ink_h0 / hu - sw) / (1 - s_ * T / hu)
        if d < 0:
            # proportional thinning: each part of a stroke loses at most 40% of its own
            # local half-width, so thin arms (e.g. the top of '[') can't vanish
            dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
            ridge = np.where((dt >= cv2.dilate(dt, np.ones((3, 3), np.float32))) & (m > 0), dt, 0).astype(np.float32)
            k = 2 * int(np.ceil(dt.max())) + 1
            local_r = cv2.dilate(ridge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            r_eff = np.minimum(-d / 2, 0.4 * local_r)
            m = ((dt > r_eff) & (m > 0)).astype(np.uint8)
            # remove hairline slivers left at stroke tips (thinner than ~30% of a stroke)
            ko = max(3, int(round(0.3 * stroke_px(m)))) | 1
            opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko)))
            if opened.sum() > 0.85 * m.sum(): m = opened
        elif d > 0:
            m = (cv2.distanceTransform((1 - m).astype(np.uint8), cv2.DIST_L2, 5) <= d / 2).astype(np.uint8)
    if char is not None and char not in COUNTERS:
        m = fill_holes(m, 1.0 if char in SINGLE_STROKE else 0.22)
    m = condense(m, xscale)
    ys, xs = np.nonzero(m)
    upp = hu / (ys.max() - ys.min() + 1)
    y_off = bottom - (m.shape[0] - (ys.max() + 1)) * upp
    contours = mask_to_contours(m, upp, y_off, NEAT.get('contour_smooth', 0.012))
    return contours, (xs.max() + 1 - xs.min()) * upp, xs.min() * upp

def synth_dot(d):
    t = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    return lambda cx, cy: [(cx + d / 2 * np.cos(a), cy - d / 2 * np.sin(a)) for a in t]  # clockwise

def cmd_build(a):
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.pens.t2CharStringPen import T2CharStringPen
    global WEIGHT, LIGHTEN, MONOLINE, CAP_WIDTH
    # project settings (font name, weights) + optional preset variant on top (e.g. --preset neat)
    cfg = {'lighten': 20, 'cap_width': 0.9, 'cap_lighten': 0, 'outdir': 'out',
           **jload(os.path.join(PRESETS, 'default.json'), {}), **jload(SETTINGS, {})}
    if a.preset:
        path = next((p for p in (os.path.join(PRESETS, a.preset + '.json'), os.path.join(PRESETS, a.preset),
                                 os.path.join(PROJ, a.preset), a.preset) if os.path.isfile(p)), None)
        if not path: sys.exit(f'preset not found: {a.preset} (see {PRESETS})')
        preset = jload(path, {})
        cfg.update({k: v for k, v in preset.items() if k != 'name_suffix'})
        cfg['name'] = cfg['name'] + preset.get('name_suffix', '')
    for key in cfg:                      # command-line flags override settings.json
        if getattr(a, key, None) is None: setattr(a, key, cfg[key])
    WEIGHT, LIGHTEN, MONOLINE, CAP_WIDTH = a.weight, a.lighten, a.monoline, a.cap_width
    print(f"settings: lighten={a.lighten} cap_width={a.cap_width} cap_lighten={a.cap_lighten}")
    global NEAT
    NEAT = cfg if cfg.get('neat') else {}
    pc = NEAT.get('prefer_copy')
    if pc == 'cleanest': pc = cleanest_copy()
    if NEAT and pc: print(f'neat: using sheet copy {pc}')
    R = ranked(prefer_copy=pc); glyphs = {}; SB = 40
    slant_fix = {}
    if NEAT.get('slant_strength'):
        lean = {c: estimate_slant(load_mask(ss[0])) for c, ss in R.items() if c in SLANT_SET}
        common = float(np.median(list(lean.values())))
        cap = NEAT.get('slant_cap_deg', 12)
        slant_fix = {c: float(np.clip(common - v, -cap, cap)) * NEAT['slant_strength'] for c, v in lean.items()}
        print(f'common slant {common:+.0f} deg; corrected', sum(abs(v) > 1 for v in slant_fix.values()), 'glyphs')
    for c, ss in R.items():
        if not c.isprintable() or c.isspace(): continue
        b, t = vmetrics(c)
        s0 = ss[0]
        if c in SYMBOL_POS:
            b, t = symbol_span(c, s0)
        if NEAT:
            if c.islower() and c in ROUND and NEAT.get('overshoot'):
                ov = NEAT['overshoot']; b, t = (b - ov if b == 0 else b), t + ov
            cn, w, x_off = build_glyph_neat(load_mask(s0), b, t, CAP_WIDTH if c.isupper() else 1.0,
                                            char=c, slant_fix=slant_fix.get(c, 0.0))
        else:
            cn, w, x_off = build_glyph(load_mask(s0), b, t, CAP_WIDTH if c.isupper() else 1.0,
                                       a.cap_lighten if c.isupper() else 0.0, char=c)
        glyphs[c] = ([[(x - x_off, y) for x, y in p] for p in cn], w)
    stroke = WEIGHT if MONOLINE else 62 - LIGHTEN / 2
    dot = synth_dot(stroke * 1.25)
    # ---- derived glyphs
    def n_parts(c):
        return cv2.connectedComponents((load_mask(R[c][0]) > 0).astype(np.uint8))[0] - 1 if c in R else 0
    if n_parts('i') < 2 and ('i' in glyphs or 'l' in glyphs):   # i: synth dot only if sample has none
        src = R['i'][0] if 'i' in R else R['l'][0]
        cn, w, x_off = build_glyph(load_mask(src), 0, XHT - 30)
        stem = [[(x - x_off, y) for x, y in p] for p in cn]
        glyphs['i'] = (stem + [dot(w / 2 + 10, XHT + 110)], w)
        if 'j' not in R and 'g' in R:
            pass
    if 'Q' not in R and 'O' in glyphs:
        cn, w = glyphs['O']
        tail = [(w * 0.55, CAP * 0.28), (w * 0.62, CAP * 0.34), (w * 1.02, -30), (w * 0.92, -50)]
        glyphs['Q'] = (cn + [tail[::-1]], w)
    for lo, up in [('z', 'Z'), ('o', 'O'), ('s', 'S'), ('c', 'C'), ('v', 'V'), ('w', 'W'), ('x', 'X'), ('k', 'K'), ('u', 'U')]:
        if up in R and lo not in R:  # fallback lowercase from capital
            cn, w, x_off = build_glyph(load_mask(R[up][0]), 0, XHT)
            glyphs[lo] = ([[(x - x_off, y) for x, y in p] for p in cn], w)
        if lo in R and up not in R:
            cn, w, x_off = build_glyph(load_mask(R[lo][0]), 0, CAP)
            glyphs[up] = ([[(x - x_off, y) for x, y in p] for p in cn], w)
    if ('j' not in glyphs or n_parts('j') < 2) and 'i' in R and n_parts('i') < 2:
        cn, w, x_off = build_glyph(load_mask(R['i'][0]), DESC + 40, XHT - 30)
        glyphs['j'] = ([[(x - x_off, y) for x, y in p] for p in cn] + [dot(w / 2 + 10, XHT + 110)], w)
    if '.' in glyphs:   # colon / semicolon built from your own full stop and comma
        pc, pw = glyphs['.']; ph = max(y for p in pc for x, y in p)
        up = [[(x, y + XHT - ph) for x, y in p] for p in pc]
        glyphs[':'] = (pc + up, pw)
        if ',' in glyphs:
            cc, cw = glyphs[',']
            dx = (cw - pw) / 2
            glyphs[';'] = (cc + [[(x + dx, y) for x, y in p] for p in up], cw)
    D = stroke * 1.2
    glyphs.setdefault('.', ([dot(D / 2, D / 2)], D))
    glyphs.setdefault(',', ([dot(D / 2, D / 2), [(D * 0.75, D * 0.3), (D * 0.45, D * 0.3), (D * 0.05, -D * 1.2), (D * 0.3, -D * 1.3)]], D))
    glyphs.setdefault(':', ([dot(D / 2, D / 2), dot(D / 2, XHT - D / 2)], D))
    glyphs.setdefault(';', (glyphs[','][0] + [dot(D / 2, XHT - D / 2)], D))
    glyphs.setdefault('-', ([[(0, 190), (0, 190 + stroke * .8), (200, 200 + stroke * .8), (200, 200)][::-1]], 200))
    glyphs.setdefault("'", ([[(D * .3, CAP), (D * .9, CAP), (D * .55, CAP - 170), (D * .3, CAP - 170)][::-1]], D))
    if '"' not in glyphs:
        q = glyphs["'"][0][0]; glyphs['"'] = ([q, [(x + D * 1.1, y) for x, y in q]], D * 2.1)
    if '!' not in glyphs and 'l' in R:
        cn, w, x_off = build_glyph(load_mask(R['l'][0]), 170, CAP)
        glyphs['!'] = ([[(x - x_off, y) for x, y in p] for p in cn] + [dot(w / 2, D / 2)], w)

    order = ['.notdef', 'space'] + [f'uni{ord(c):04X}' for c in sorted(glyphs)]
    fb = FontBuilder(UPM, isTTF=not a.otf)
    fb.setupGlyphOrder(order)
    cmap = {32: 'space'}; cmap.update({ord(c): f'uni{ord(c):04X}' for c in glyphs})
    fb.setupCharacterMap(cmap)
    metrics = {'.notdef': (500, 0), 'space': (300, 0)}
    def notdef(pen, otf):
        boxes = [[(50, 0), (50, 700), (450, 700), (450, 0)], [(100, 50), (400, 50), (400, 650), (100, 650)]]
        for box in boxes:
            box = box[::-1] if otf else box
            pen.moveTo(box[0]); [pen.lineTo(q) for q in box[1:]]; pen.closePath()
        return pen
    if a.otf:
        cs = {'.notdef': notdef(T2CharStringPen(500, None), True).getCharString(),
              'space': T2CharStringPen(300, None).getCharString()}
    else:
        ttg = {'.notdef': notdef(TTGlyphPen(None), False).glyph(), 'space': TTGlyphPen(None).glyph()}
    for c, (cn, w) in glyphs.items():
        sb = SB * 0.75 if NEAT.get('optical_sb') and c in ROUND else SB   # round shapes look farther apart
        name = f'uni{ord(c):04X}'; adv = int(round(w + 2 * sb))
        pen = T2CharStringPen(adv, None) if a.otf else TTGlyphPen(None)
        for p in cn:
            pts = [(int(round(x + sb)), int(round(y))) for x, y in p]
            pts = [q for j, q in enumerate(pts) if q != pts[j - 1]]   # no duplicate points
            if len(pts) < 3: continue
            if a.otf: pts = pts[::-1]  # CFF uses counter-clockwise outers
            pen.moveTo(pts[0])
            for q in pts[1:]: pen.lineTo(q)
            pen.closePath()
        if a.otf: cs[name] = pen.getCharString()
        else: ttg[name] = pen.glyph()
        metrics[name] = (adv, int(round(sb)))
    if a.otf: fb.setupCFF(a.name.replace(' ', '') + '-Regular', {'FullName': a.name + ' Regular'}, cs, {})
    else: fb.setupGlyf(ttg)
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=ASC + 80, descent=DESC - 50)
    ps = a.name.replace(' ', '') + '-Regular'
    fb.setupNameTable({'familyName': a.name, 'styleName': 'Regular', 'uniqueFontIdentifier': f'1.000;NONE;{ps}',
                       'fullName': f'{a.name} Regular', 'psName': ps, 'version': 'Version 1.000',
                       'copyright': 'Personal handwriting font'})
    fb.setupOS2(sTypoAscender=ASC + 80, sTypoDescender=DESC - 50, sTypoLineGap=150,
                usWinAscent=ASC + 150, usWinDescent=-DESC + 100, sxHeight=XHT, sCapHeight=CAP)
    fb.setupPost()
    from fontTools.ttLib.removeOverlaps import removeOverlaps
    removeOverlaps(fb.font)          # merge overlapping/self-intersecting contours, fix direction
    os.makedirs(OUT, exist_ok=True)
    outdir = os.path.join(PROJ, a.outdir); os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, a.name.replace(' ', '') + ('.otf' if a.otf else '.ttf'))
    fb.save(path); print('wrote', path, f'({len(glyphs)} glyphs)')
    if not a.otf:   # web font alongside the TrueType file (needs `brotli`)
        try:
            from fontTools.ttLib import TTFont
            wf = TTFont(path); wf.flavor = 'woff2'; wf.save(path[:-4] + '.woff2'); print('wrote', path[:-4] + '.woff2')
        except ImportError:
            print('(skipped .woff2: pip install brotli)')
    missing = [c for c in map(chr, range(33, 127)) if c not in glyphs]
    print('missing:', ''.join(missing) or 'none')
    return path

def cmd_specimen(a):
    from PIL import Image, ImageDraw, ImageFont
    f = ImageFont.truetype(a.font, 72); fs = ImageFont.truetype(a.font, 44)
    lines = [('ABCDEFGHIJKLMNOPQRSTUVWXYZ', f), ('abcdefghijklmnopqrstuvwxyz', f), ('0123456789 .,:;-\'"?!/', f), ('()[]{}<>@#$%&*+=^~`|\\ ₹', f),
             ('', fs), ('The quick brown fox jumps over the lazy dog.', fs),
             ('"Sphinx of black quartz, judge my vow!"', fs),
             ('Meeting notes - Q4 planning - 24/11/2026', fs),
             ('Price: ₹1,250 (incl. 18% GST) & free shipping @ home!', fs)]
    img = Image.new('L', (1900, 110 * len(lines) + 60), 255); d = ImageDraw.Draw(img); y = 30
    for t, ft in lines:
        d.text((40, y), t, font=ft, fill=0); y += 110 if ft is f else 80
    p = os.path.join(os.path.dirname(a.font), 'specimen.png'); img.save(p); print(p)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--project', help='folder for YOUR handwriting (data, fonts, settings); '
                                      'default: the example project in this repo')
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('ingest'); p.add_argument('paths', nargs='+'); p.add_argument('--force', action='store_true'); p.set_defaults(f=cmd_ingest)
    p = sp.add_parser('ingest-sheet', help='ingest a character sample sheet written per a layout file')
    p.add_argument('image'); p.add_argument('--layout', default=os.path.join(ROOT, 'layouts', 'charsheet.txt'))
    p.add_argument('--copies', type=int, default=1, help='how many times the layout is written on the sheet')
    p.add_argument('--rotate', choices=['cw', 'ccw', '180']); p.add_argument('--name'); p.set_defaults(f=cmd_ingest_sheet)
    p = sp.add_parser('reindex'); p.set_defaults(f=lambda a: reindex())
    p = sp.add_parser('region'); p.add_argument('page')
    for k in ('x0', 'y0', 'x1', 'y1'): p.add_argument(k, type=int)
    p.add_argument('--rot', choices=['cw', 'ccw']); p.add_argument('--tag', required=True); p.set_defaults(f=cmd_region)
    p = sp.add_parser('label'); p.add_argument('items', nargs='+'); p.set_defaults(f=cmd_label)
    p = sp.add_parser('sheet'); p.add_argument('chars', nargs='?'); p.add_argument('--out'); p.set_defaults(f=cmd_sheet)
    p = sp.add_parser('pick'); p.add_argument('char'); p.add_argument('id'); p.set_defaults(f=cmd_pick)
    p = sp.add_parser('reject'); p.add_argument('ids', nargs='+'); p.set_defaults(f=cmd_reject)
    p = sp.add_parser('build'); p.add_argument('--preset', '--settings', dest='preset', help='variant preset on top of the project settings, e.g. "neat"'); p.add_argument('--name', default=None); p.add_argument('--otf', action='store_true'); p.add_argument('--lighten', type=float, default=None, help='units shaved off stroke thickness (settings.json)'); p.add_argument('--cap-width', type=float, default=None, help='horizontal scale for capitals (settings.json)'); p.add_argument('--cap-lighten', type=float, default=None, help='extra thinning for capitals in font units (settings.json)'); p.add_argument('--monoline', action='store_true', help='uniform pen redraw (does not preserve hand look)'); p.add_argument('--weight', type=float, default=100, help='pen thickness for --monoline')
    p.set_defaults(f=lambda a: cmd_specimen(argparse.Namespace(font=cmd_build(a))))
    a = ap.parse_args()
    set_project(a.project or ROOT)
    a.f(a)

if __name__ == '__main__':
    main()
