# PDF Tamper Detector — Synthetic Test Kit

A labeled set of "original" vs "edited" bank-statement PDFs, plus a working
detector that scores each file on three independent forensic signals and
combines them into one confidence number. Built because real bank
statements aren't something you can legally scrape or download for testing
— see the note at the bottom.

## Folder layout

```
generators/
  gen_text_original.py   -> clean, native-text PDF statement
  gen_text_edited.py     -> same file, tampered via a REAL incremental
                             update (appends a forged revision, doesn't
                             touch the original bytes)
  gen_scan_original.py   -> "scanned" statement: text rendered to an image,
                             saved as JPEG, wrapped losslessly in a PDF
  gen_scan_edited.py     -> same scan, one number edited on the pixels and
                             re-saved at a different JPEG quality
detector/
  detector.py             -> the actual analyzer: run it on any PDF(s)
samples/                  -> the four generated PDFs + renders, already run
```

## Run it

```bash
pip install pypdf reportlab pillow numpy scipy img2pdf --break-system-packages
python3 generators/gen_text_original.py
python3 generators/gen_text_edited.py
python3 generators/gen_scan_original.py
python3 generators/gen_scan_edited.py
python3 detector/detector.py samples/*.pdf
```

## Results on the labeled set

| file | confidence edited | verdict |
|---|---|---|
| text_original.pdf | 0% | LIKELY ORIGINAL |
| text_edited.pdf | 100% | LIKELY EDITED |
| scan_original.pdf | 15% | LIKELY ORIGINAL |
| scan_edited.pdf | 81% | LIKELY EDITED |

## The three checks

**1. Metadata (`check_metadata`)** — pulls the PDF Info dictionary via
`pypdf` and flags: ModDate meaningfully after CreationDate, and a
Producer that reads like a generic editing tool (Acrobat, Photoshop, GIMP,
Word, etc.) while Creator still shows the "original" generator. This is
exactly what happens in real life: most editors update Producer on
resave but don't bother touching Creator.

**2. Revision history (`check_revisions`)** — looks for a `/Prev` key in
a trailer dictionary, which is the spec-correct signature of a PDF
*incremental update* (most editors append changes rather than rewriting
the whole file, to keep save times fast). `text_edited.pdf` was built by
hand-crafting exactly this: the original bytes are 100% untouched, a
forged content stream + Info dict are appended after them, with a new
xref/trailer pointing `/Prev` back at the original.

**3. Error Level Analysis (`check_ela`)** — for PDFs with embedded raster
images (the scanned-statement case), re-saves each image at a fixed test
quality and diffs it against itself. A patch that was freshly rendered and
recompressed tends to sit several MADs above the image's own block-error
baseline. Score is based on that local anomaly (`local_z`), not on
comparing absolute error levels *between* files — more on why below.

## Combining the three scores

The three checks are combined as a **probabilistic OR**
(`confidence = 1 - (1-m)(1-r)(1-e)`), not a weighted average. This
matters: `scan_edited.pdf` has completely clean metadata and no revision
trail (a competent forger nailed both), but a strong ELA hotspot. A plain
average would have diluted that one real signal down to ~28% — below any
sane threshold — just because the other two channels came back clean.
With OR-combination it correctly lands at 81%. Real fraud-scoring systems
use this same logic: one strong independent red flag should dominate, not
get averaged away by channels that simply don't apply to that forgery
method.

## Bugs found and fixed while building this (worth mentioning to your supervisor)

- **`reportlab`'s `drawImage` silently recompresses embedded images** at
  its own default JPEG quality — any quality setting you chose when
  saving the source JPEG gets thrown away. Switched to `img2pdf`, which
  embeds JPEG bytes byte-for-byte.
- **`pypdf`'s `page.images[i].data` decodes and re-encodes** the image via
  PIL before handing it back — same problem, one layer up. Fixed by
  pulling the raw `/DCTDecode` stream via `.get_data()` directly, which
  for an already-JPEG image returns the untouched compressed bytes.
- **Naive `%%EOF` byte-scanning gives false positives two ways**: (a) that
  5-byte sequence can coincidentally appear inside binary image/font
  data, and (b) a **linearized ("fast web view") PDF legitimately has a
  second trailer with `/Prev`** even though it was never edited after
  creation — `img2pdf` does this by default via its pikepdf backend. Fixed
  by forcing `engine=img2pdf.Engine.internal` in the generators, and by
  noting in the detector's own output that a `/Prev` hit should be
  cross-checked against metadata/content, not trusted alone.

## Honest limitations

- **ELA is much noisier on text-dense scans than on photographs.** Text
  edges are naturally high-frequency and produce elevated block error
  everywhere, not just at a tamper site — this is a known, documented
  weakness of ELA, not a bug in this implementation. On a real scanned
  statement, expect more false hotspots than on a clean photo. Consider
  pairing it with copy-move detection (`cv2.ORB`/`cv2.SIFT` keypoint
  matching) for a second, independent pixel-level signal.
- **The revision-history check can false-positive on linearized PDFs.**
  Treat a `/Prev` hit as "resaved at some point," not automatically
  "edited with fraudulent intent" — plenty of legitimate PDF workflows
  (print-to-PDF, "optimize for web" in Acrobat) linearize without any
  content change.
- **Thresholds here (0.4 verdict cutoff, the ELA calibration constants)
  are tuned against exactly 4 labeled files.** Before trusting this on
  real documents, generate many more synthetic pairs (vary account
  numbers, tamper types, JPEG qualities, editors) and recalibrate.

## Where the "data" actually comes from

There's no legitimate downloadable dataset of real people's bank
statements you could ethically use for this — anything claiming to be
that is either fabricated for loan-fraud purposes or leaked personal
data, and neither belongs in a university project. Generating your own
synthetic originals + tampered pairs (what this kit does) is the actual
standard approach in document-forensics research, because it's the only
way to get labeled ground truth. If you want more realistic scanned-
document *textures/layouts* to test robustness (not fraud-labeled, but
real and free to use): **SROIE** and **FUNSD** (scanned receipts/forms),
**RVL-CDIP** (scanned document classification).
