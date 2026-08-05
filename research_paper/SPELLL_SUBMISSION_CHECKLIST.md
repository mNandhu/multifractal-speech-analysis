# SPELLL 2026 submission checklist

Target: Springer CCIS (LNCS class), **double-blind**, OpenReview.
CFP: https://spelll.org/callforpapers.html
Submission link: https://openreview.net/group?id=SPELLL.org/2026/Conference

Source file: `paper_ccis.tex` (converted from `paper.tex`, IEEEtran).
`llncs.cls` is vendored in this directory, so no extra install is needed.

```
pdflatex -interaction=nonstopmode -output-directory=tmp_latexv2_out paper_ccis.tex   # run twice
```

Current build: **14 pages**, 0 errors, 0 overfull boxes. Regular-paper band is 12–16.

---

## What changed from `paper.tex` → `paper_ccis.tex`

Prose is byte-identical. `diff` over the region between `\section{Introduction}`
and `\begin{thebibliography}` shows only float environments and table column
specs — no sentence was touched.

### Document class / preamble
- `IEEEtran` (`conference`) → `\documentclass[runningheads]{llncs}`.
- Dropped `\IEEEoverridecommandlockouts`, `cite`, `listings` (+ `\lstset`),
  `longtable`, and the unused `\def\BibTeX` — all IEEE-only or dead.
- Added `\usepackage[T1]{fontenc}` (Springer requirement) and `rotating`-free
  layout; kept `amsmath`, `graphicx`, `textcomp`, `booktabs`, `multirow`, `array`.
- `\urlstyle{rm}` + blue `\UrlFont` per Springer eBook style.
- `\emergencystretch=2em` — the single-column LNCS measure (12.2 cm) is much
  narrower than the IEEE column pair, and two paragraphs went overfull without it.

### Header
- `\titlerunning{}` and `\authorrunning{}` added — required by `runningheads`.
- `\begin{IEEEkeywords}` → `\keywords{A \and B \and ...}` inside `abstract`.
  Abstract is 173 words (LNCS wants 150–250).

### Floats
- LNCS is single-column, so `figure*` is invalid: all four starred floats →
  plain `figure`.
- Four figures were drawn for a two-column span and became illegible at
  12.2 cm. Wide multi-panel images are now split with `trim`/`clip`:
  - `binary_confusion_models` (4 panels in a row) → 2×2 grid.
  - `multiclass_confusion` (3 panels) → 2 + 1.
  - `feature_ablation` (2 panels) → stacked, full width each.
  - `data_pipeline` is a single linear flow diagram; left at `\textwidth`.
- Crop offsets are the **detected subplot gutters**, not `width/N`. Matplotlib
  leaves asymmetric outer margins, so equal-fraction cuts drift a few pixels
  per panel and the grid rows visibly stagger. Gutters were found by scanning
  for all-white columns; the resulting panel content widths are exactly equal
  (388 / 433 / 923 px), so the rows line up. Re-derive these numbers if any
  figure is regenerated.
- `\topfraction` 0.7 → 0.9 (plus `bottomfraction`, `textfraction`,
  `floatpagefraction`) and `[tb]` → `[htbp]`. The taller split figures exceed
  the default top-of-page allowance and were otherwise stranded *after* the
  bibliography.
- Table captions were already above the tabular, which matches LNCS.
- `p{3.2cm} p{2cm}` → `L{5.4cm} L{2.4cm}` in Tables 4 and 5, using the
  `L` raggedright column already defined (and previously unused) in the
  preamble. Removes cell wrapping and the underfull boxes it caused.

### Bibliography
- The hand-written `thebibliography` block is **gone**. All 20 entries now live
  in `refs.bib`; `paper_ccis.tex` ends with `\bibliographystyle{splncs04}` +
  `\bibliography{refs}`, which is the path `ccis_template.tex:147` prescribes.
  Build: `pdflatex` → `bibtex` → `pdflatex` ×2. BibTeX exits with 0 warnings.
- Overleaf builds this as-is: `splncs04.bst` is already in its TeX Live, so
  only `paper_ccis.tex`, `refs.bib`, `llncs.cls` and `figures/` need uploading,
  and latexmk runs BibTeX for you. Confirmed working.
- `.bbl`/`.pdf` are build output and are **not** committed. For the Springer
  camera-ready source package, regenerate `paper_ccis.bbl` and include it —
  without it a single `pdflatex` pass yields an empty reference list.
- `splncs04.bst` sorts alphabetically by first author and lowercases titles to
  sentence case. Numbers therefore do **not** follow order of first citation
  (the first citation in the text is `[17]`, Sáenz-Lechón) — correct for
  LNCS/CCIS, unlike IEEE, where `IEEEtran.bst` numbers by order of appearance.
- Two `.bib` details that exist only to satisfy the sort/case rules:
  `svd` carries `key = {Saarbrucken}` (an author-less entry otherwise has no
  sort key and BibTeX dumps it at position 1 with a warning), and proper nouns
  are brace-protected (`{GeMAPS}`, `{Parkinson's}`, `{Dysphonia Severity Index}`)
  so `change.case$` does not lowercase them.

### Anonymization (double-blind)
- All five author names, the Amrita affiliation, the city, and all five e-mail
  addresses removed. `\author{Anonymous Author(s)}`,
  `\institute{Affiliation withheld for double-blind review}`.
- The stray empty `\IEEEauthorblockN{ }` and the `\hspace{5cm}` layout hacks
  are gone with the author block.
- `\hypersetup{pdfauthor={},pdfsubject={},pdfkeywords={}}` so no name leaks
  into PDF metadata. Verified: `author` and `title` fields are empty.
- Full-text scan of the built PDF for author surnames, `Amrita`,
  `Vidyapeetham`, `Coimbatore`, `amrita.edu`, `cb.ai` → no hits.
- No first-person self-citation phrasing anywhere; `lal2020multifractal`,
  `krishnan2026` and `jahnavi2020` are already cited in the third person.

---

## Still your call

- [ ] **`krishnan2026`** is cited as *IEEE Access* vol. 14, **2026** and shares
      co-authors with this paper. A reviewer who searches it can de-anonymize
      the submission. Either confirm it is publicly indexed, or drop the
      forward-dated volume.
- [ ] **Conclusion says "macro F1 of 0.89"; Table 2 says 0.882.** Inherited
      from `paper.tex`; not touched because you asked for format-only changes.
- [x] **Uncited references.** Resolved: `ververidis2006` and `alhanai2018` were
      dropped; `svd` and `opensmile` are cited. All 20 entries are now cited.
- [ ] **AI disclosure.** SPELLL follows Springer's AI policy. Put the statement
      in the camera-ready, not the blind submission.
- [ ] **Track.** Track 3 (Speech Technologies) — "Speech, voice, and hearing
      disorders". Track 5 lists "Disordered Speech with NLP" as an alternative.
- [ ] **Register + present.** At least one author must register by the early date.

## If you need to move the page count

Currently 12 (verified from the build log). Levers that do not touch prose:

- Down: revert the three figure splits to single `\includegraphics` at
  `\textwidth` → 11 pages (but the panels become unreadable, and 11 is below
  the 12-page floor).
- Up: `figures/` still holds four unused images —
  `pathology_distribution.png`, `waveform_spectrogram_examples.png`,
  `multifractal_feature_distributions.png`, `age_impact.png`. Adding any of
  them is a content change and needs new caption text.
