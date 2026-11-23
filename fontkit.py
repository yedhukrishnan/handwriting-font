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
OUT = os.path.join(ROOT, 'out')
REVIEW = os.path.join(OUT, 'review')

def jload(p, d):
    return json.load(open(p)) if os.path.exists(p) else d
def jsave(p, o):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(o, open(p, 'w'), indent=1, ensure_ascii=False, default=lambda v: v.item())

def safe(c):  # directory/file-safe name for a char
    return f'U+{ord(c):04X}'

# ---------------------------------------------------------------- binarize
def binarize(path):
    img = cv2.imread(path)
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
                            if f.lower().endswith(('.jpg', '.jpeg', '.png')))
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
    the private journal pages are git-ignored) are carried over unchanged from the
    existing library instead of being dropped.
    """
    have_page = lambda name: os.path.exists(os.path.join(PAGES, name + '.png'))
    regs = jload(REGIONS, {})
    def source_page(entry):
        if entry['src'] == 'manual':
            return regs.get(entry['id'][2:].rsplit(':', 1)[0], {}).get('page', '')
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

def ranked():
    """char -> samples best-first: picks, then manual labels, then auto by typicality."""
    idx = jload(INDEX, []); sel = jload(SEL, {'pick': {}, 'reject': []})
    rej = set(sel['reject']); by = collections.defaultdict(list)
    for s in idx:
        if s['id'] not in rej: by[s['char']].append(s)
    res = {}
    for c, ss in by.items():
        auto = [s for s in ss if s['src'] == 'auto']
        if len(auto) >= 3:
            V = np.array([norm_vec(os.path.join(DATA, s['file'])) for s in auto]); med = np.median(V, 0)
            d = ((V - med) ** 2).sum(1); auto = [auto[i] for i in np.argsort(d)]
        man = [s for s in ss if s['src'] == 'manual']
        pick = sel['pick'].get(c)
        res[c] = sorted(man + auto, key=lambda s: s['id'] != pick)
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
ASCENDERS = set('bdfhklt') | set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789?!&/')
DESCENDERS = set('gjpqy')

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
    big = cv2.resize(m.astype(np.float32), None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
    sig = max(1.0, 1.2 * sc)
    big = cv2.GaussianBlur(big, (0, 0), sig)
    out = (big > 127).astype(np.uint8)
    # fill pin-holes left by uneven ink, keep real counters (o, e, a ...)
    cnts, hier = cv2.findContours(out, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    for i, cn in enumerate(cnts):
        if hier[0][i][3] >= 0 and cv2.contourArea(cn) < (0.004 * out.size):
            cv2.drawContours(out, [cn], -1, 1, -1)
    return out, 4 * sc

def mask_to_contours(mask, units_per_px, y_off):
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    res = []
    if hier is None: return res
    h = mask.shape[0]
    for i, cn in enumerate(cnts):
        if cv2.contourArea(cn) < 30: continue
        ap = cv2.approxPolyDP(cn, 0.8, True).reshape(-1, 2)
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
STROKE_PX = 7.6   # typical pen stroke width in source pixels (median over lowercase samples)

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

def build_glyph(mask, bottom, top):
    """Natural outline of the handwritten glyph, evenly thinned by LIGHTEN units.

    Thinning peels the same amount off every edge, so the pen's natural thick/thin
    variation is preserved. The glyph is scaled so that after thinning it still spans
    exactly bottom..top.
    """
    if MONOLINE:
        return build_glyph_monoline(mask, bottom, top)
    hu = top - bottom
    m, pad = smooth_mask(mask, 220)
    ink_h = m.shape[0] - 2 * pad
    upp = (hu + LIGHTEN) / ink_h
    if LIGHTEN > 0:
        r = LIGHTEN / 2 / upp
        thin = (cv2.distanceTransform(m, cv2.DIST_L2, 5) > r).astype(np.uint8)
        if thin.sum() > 0.3 * m.sum():      # never let thin glyphs disappear
            m = thin
    contours = mask_to_contours(m, upp, bottom - LIGHTEN / 2 - pad * upp)
    return contours, (m.shape[1] - 2 * pad) * upp - LIGHTEN, -pad * upp + LIGHTEN / 2

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

def synth_dot(d):
    t = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    return lambda cx, cy: [(cx + d / 2 * np.cos(a), cy - d / 2 * np.sin(a)) for a in t]  # clockwise

def cmd_build(a):
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.pens.t2CharStringPen import T2CharStringPen
    global WEIGHT, LIGHTEN, MONOLINE
    WEIGHT, LIGHTEN, MONOLINE = a.weight, a.lighten, a.monoline
    R = ranked(); glyphs = {}; SB = 40
    for c, ss in R.items():
        if not (c.isascii() and c.isprintable()) or c == ' ': continue
        b, t = vmetrics(c)
        cn, w, x_off = build_glyph(load_mask(ss[0]), b, t)
        glyphs[c] = ([[(x - x_off, y) for x, y in p] for p in cn], w)
    stroke = WEIGHT if MONOLINE else 62 - LIGHTEN / 2
    dot = synth_dot(stroke * 1.25)
    # ---- derived glyphs
    if 'i' in glyphs or 'l' in glyphs:   # i/j: dotless stem + dot (OCR rarely keeps the dot)
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
    if 'j' not in glyphs and 'i' in R:
        cn, w, x_off = build_glyph(load_mask(R['i'][0]), DESC + 40, XHT - 30)
        glyphs['j'] = ([[(x - x_off, y) for x, y in p] for p in cn] + [dot(w / 2 + 10, XHT + 110)], w)
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
        name = f'uni{ord(c):04X}'; adv = int(round(w + 2 * SB))
        pen = T2CharStringPen(adv, None) if a.otf else TTGlyphPen(None)
        for p in cn:
            pts = [(int(round(x + SB)), int(round(y))) for x, y in p]
            pts = [q for j, q in enumerate(pts) if q != pts[j - 1]]   # no duplicate points
            if len(pts) < 3: continue
            if a.otf: pts = pts[::-1]  # CFF uses counter-clockwise outers
            pen.moveTo(pts[0])
            for q in pts[1:]: pen.lineTo(q)
            pen.closePath()
        if a.otf: cs[name] = pen.getCharString()
        else: ttg[name] = pen.glyph()
        metrics[name] = (adv, SB)
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
    path = os.path.join(OUT, a.name.replace(' ', '') + ('.otf' if a.otf else '.ttf'))
    fb.save(path); print('wrote', path, f'({len(glyphs)} glyphs)')
    missing = [c for c in map(chr, range(33, 127)) if c not in glyphs]
    print('missing:', ''.join(missing) or 'none')
    return path

def cmd_specimen(a):
    from PIL import Image, ImageDraw, ImageFont
    f = ImageFont.truetype(a.font, 72); fs = ImageFont.truetype(a.font, 44)
    lines = [('ABCDEFGHIJKLMNOPQRSTUVWXYZ', f), ('abcdefghijklmnopqrstuvwxyz', f), ('0123456789 .,:;-\'"?!/', f),
             ('', fs), ('The quick brown fox jumps over the lazy dog.', fs),
             ('"Making sense of one\'s life as a whole?"', fs),
             ('Reset your mind, body and spirit - 18/03/2025', fs)]
    img = Image.new('L', (1900, 110 * len(lines) + 60), 255); d = ImageDraw.Draw(img); y = 30
    for t, ft in lines:
        d.text((40, y), t, font=ft, fill=0); y += 110 if ft is f else 80
    p = os.path.join(OUT, 'specimen.png'); img.save(p); print(p)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('ingest'); p.add_argument('paths', nargs='+'); p.add_argument('--force', action='store_true'); p.set_defaults(f=cmd_ingest)
    p = sp.add_parser('reindex'); p.set_defaults(f=lambda a: reindex())
    p = sp.add_parser('region'); p.add_argument('page')
    for k in ('x0', 'y0', 'x1', 'y1'): p.add_argument(k, type=int)
    p.add_argument('--rot', choices=['cw', 'ccw']); p.add_argument('--tag', required=True); p.set_defaults(f=cmd_region)
    p = sp.add_parser('label'); p.add_argument('items', nargs='+'); p.set_defaults(f=cmd_label)
    p = sp.add_parser('sheet'); p.add_argument('chars', nargs='?'); p.add_argument('--out'); p.set_defaults(f=cmd_sheet)
    p = sp.add_parser('pick'); p.add_argument('char'); p.add_argument('id'); p.set_defaults(f=cmd_pick)
    p = sp.add_parser('reject'); p.add_argument('ids', nargs='+'); p.set_defaults(f=cmd_reject)
    p = sp.add_parser('build'); p.add_argument('--name', default='Yedhu'); p.add_argument('--otf', action='store_true'); p.add_argument('--lighten', type=float, default=20, help='units shaved off stroke thickness (default 20; 0 = original ink)'); p.add_argument('--monoline', action='store_true', help='uniform pen redraw (does not preserve hand look)'); p.add_argument('--weight', type=float, default=100, help='pen thickness for --monoline')
    p.set_defaults(f=lambda a: cmd_specimen(argparse.Namespace(font=cmd_build(a))))
    a = ap.parse_args(); a.f(a)

if __name__ == '__main__':
    main()
