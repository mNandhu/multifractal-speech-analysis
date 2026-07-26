# SPELLL 2026 submission checklist

Target: Springer CCIS (LNCS class), **double-blind**, OpenReview.
Submission link: https://openreview.net/group?id=SPELLL.org/2026/Conference

---

## Before you can compile

`paper_ccis.tex` needs `llncs.cls`, which does **not** ship with MiKTeX or TeX Live.

1. Download the Springer LaTeX zip:
   https://resource-cms.springernature.com/springer-cms/rest/v1/content/19238648/data/v8
2. Unzip and copy `llncs.cls` into `research_paper/`.
   (`splncs04.bst` is not needed — the bibliography is a manual `thebibliography`
   already formatted in splncs04 style.)
3. `pdflatex paper_ccis.tex` — run twice so cross-references resolve.

Overleaf alternative: https://www.overleaf.com/latex/templates/springer-lecture-notes-in-computer-science/kzwwpvhwnvfj

---

## What changed from `paper.tex` (IEEE) → `paper_ccis.tex` (CCIS)

### Format
- `IEEEtran` → `\documentclass[runningheads]{llncs}`; dropped `\IEEEoverridecommandlockouts`, `cite`, `listings`, `longtable`.
- Added `\titlerunning{}` and `\authorrunning{}` (required by `runningheads`).
- `\begin{IEEEkeywords}` → `\keywords{A \and B \and C}` inside the abstract environment.
- All six `figure*` / `table*` starred floats → plain `figure` / `table`. LNCS is
  single-column, so starred floats are invalid.
- Figure widths reduced from `\textwidth` where the original was sized for a
  two-column span (`pathology_distribution` → 0.85, `age_impact` → 0.9).
- Table captions moved above the tabular (LNCS convention).
- All 24 bibliography entries reformatted from IEEE style to splncs04 style
  (`Surname, F.M.: Title. Journal 12(3), 45--67 (2020)`).

### Anonymization
- Removed all five author names, the Amrita affiliation, and all five email
  addresses; replaced with `\author{Anonymous Author(s)}` /
  `\institute{Anonymous Institution}`.
- Removed the stray empty `\IEEEauthorblockN{ }` and the `\hspace{5cm}` layout hacks.
- Replaced `\texttt{StratifiedGroupKFold}` with the neutral phrase "stratified grouped
  $k$-fold" — a bare scikit-learn API name is harmless, but the original wording read
  like a code walkthrough.
- All self-citations already use third person; no "our earlier work" phrasing anywhere.

### Content added (to clear the 12-page regular-paper minimum)
- **New Section 3, "Multifractal Formulation"** — the full MFDFA procedure with
  9 numbered equations (profile, segmentation, local variance ×2, $F_q(s)$, $F_0(s)$,
  scaling, $\tau(q)$, Legendre transform, $\Delta\alpha$, asymmetry), a table
  documenting all 12 derived descriptors, and a parameter-selection justification.
  Descriptors and parameters were read from `src/features/feature_extraction.py`
  so they match the code exactly.
- **New Section 8, "Discussion"** — four subsections: multifractal features as a
  standalone representation, the age confound (both readings, honestly stated),
  why disease-group classification is harder, and explicit limitations.
- Related Work split into five labelled subsections.
- Added a paper-organisation paragraph at the end of the Introduction.
- Added the four previously unused figures: `pathology_distribution.png`,
  `waveform_spectrogram_examples.png`, `multifractal_feature_distributions.png`,
  `age_impact.png`.
- Named the four classifiers and stated the fold-wise standardisation protocol
  (previously unstated — a reviewer would have asked).
- Expanded the per-disease failure analysis for vocal fold nodules and spasmodic
  dysphonia.

### Unverified claims — removed in revision

The first draft contained assertions I inferred rather than checked. Resolved as follows:

| Claim | Location | Resolution |
|---|---|---|
| "Δα and α₀ show the clearest separation" | §5.1 | **Removed.** Figure is now referenced neutrally. |
| "The descriptors most responsible are Δα and α₀" | §8.1 | **Removed**, replaced with an explicit statement that feature-importance analysis is future work. |
| "SVD nodule cohort skews toward younger female speakers" | §7.4 | **Removed.** Replaced with a small-sample-size caveat, which is defensible from the data. |
| "low-resource clinical settings where laryngological expertise is scarce" | §1 | **Removed.** SVD is a German university clinical corpus; the framing did not describe the data. |
| "standardised to zero mean and unit variance" | §6 | **Corrected** against `model_training_v7.ipynb`: Yeo–Johnson power transform with standardisation, linear model only; trees receive raw numerics with NaN preserved. Balanced class weighting added. |
| `ℓ2` logistic regression | §6 | **Verified correct** — sklearn default penalty, `C=1.0`, `max_iter=3000`, `class_weight='balanced'`. |

Resolved in the second pass:

- **§2.4, ivanov1999** — verified against the source. Reworded to "a loss of multifractality
  accompanies congestive heart failure," which is the paper's actual stated finding.
- **`ali2016`** — verified. J. Med. Syst. **40**(1), article 20 (2016), DOI
  10.1007/s10916-015-0392-2. Issue number added.
- **"all reported experiments use only `a_n`"** — verified. `model_training_v7`,
  `model_training_per_disease`, and `feature_ablation_study` all set
  `SELECTED_TOKEN='a_n'`. The multi-token notebooks (`v3_grouped`, `v3_1_grouped`,
  `v4_sample_level`) set it to `None` but none of their numbers appear in the paper.
- **MFDFA parameters** — verified against `data/processed/features/feature_build_config.json`:
  `q ∈ [-5, 5]`, step `0.5`, `40` scales, order `1`. Note these override the dataclass
  defaults in `feature_options.py` (step `1.0`, `20` scales), so the config file is the
  authoritative record.

Still standing, your call:

- **§8.4, Limitations** — "The poor per-disease result for spasmodic dysphonia is likely
  attributable to this choice [of sustained-vowel token]." Hedged conjecture in a
  Limitations section, which is where conjecture belongs. Delete if you disagree.

### Errors found by fact-checking (fixed)

Verification against the notebooks turned up four substantive problems, three of them
inherited from `paper.tex` and one I introduced.

1. **The disease-group task is three-class, not two.** `model_training_v7.ipynb` shows
   `target_label` = healthy 478 / Neurological 277 / Structural 201. Your original abstract
   called it "multi-class" and was right; my first pass "corrected" this to a two-class
   framing, which was wrong. Reverted and made explicit throughout, with the group
   composition and per-class counts now stated in §7.2.
2. **Per-disease experiments did not use five folds.** The notebook reduces splits to two
   for every condition ("not enough mixed-class speakers"). Both `paper.tex` and my first
   pass claimed "all experiments employ five-fold" cross-validation. §7 now scopes the
   five-fold claim to the binary, three-class, and ablation experiments, and
   Section 7.4 states the two-fold protocol.
3. **Table 6 omitted cohort sizes, and one row is meaningless without them.** Vocal fold
   nodules is n=17 per class over two folds; 0.531 balanced accuracy is indistinguishable
   from chance. An `n/class` column has been added to the table and the text now says so
   directly. Also added: the healthy pool is downsampled 1:1 per condition, so chance is
   0.5 in every row.
4. **Reinke's edema appears in Table 6 but not in the Dataset section's disorder list.**
   The per-disease experiment draws from a broader 13-disease config than the main tasks.
   §7.4 now notes this rather than leaving the tables inconsistent.

Also corrected: the ablation's "MFDFA only" configuration retains age and sex (as the
ablation text already stated), so describing 0.866 as the multifractal descriptors "in
isolation" was wrong. Abstract, §8.1, and the conclusion now say "without conventional
acoustic descriptors" and §8.1 cross-references the age caveat.

### Removed as unsourceable

- §2.3 — "A recurring observation across these studies is that performance degrades
  sharply as the label space moves toward fine-grained aetiological categories."
  Characterisation of three cited papers I did not re-read.
- §2.5 — "a substantial fraction of published results do not state it." Unverified
  quantitative claim about the literature.
- §2.5 — "pathological cohorts in clinical corpora are typically older than healthy
  control cohorts." Rewritten as a conditional about this corpus rather than a general fact.
- §5.3 — "These serve partly as a control condition." Asserted intent behind your feature
  choice; rewritten to describe what the comparison shows instead.

### Corrections
- Conclusion said "macro F1 of **0.89**"; Table 5 says 0.882. Fixed to 0.882.
- Table 3 bolded Random Forest's accuracy but LightGBM has the higher balanced
  accuracy and macro F1; bolding now follows the column maxima.
- Abstract said "multi-class (neurological/structural disease group)" — two classes
  is binary, not multi-class. Reworded to "disease-group".
- `ali2016` was missing an article number; added `20`.

---

## Action items for you

- [ ] **Download `llncs.cls`** and compile. Report the page count back.
- [ ] **Verify `krishnan2026`.** Cited as *IEEE Access* vol. 14, **2026** — i.e. in press
      or very recent. It shares a co-author with your last author. If it is not yet
      publicly indexed, a reviewer who searches it may de-anonymize you. Options:
      confirm it is published and leave it, or cite it as "recent work" without the
      forward-dated volume.
- [ ] **Check PDF metadata.** `pdflatex` can embed the author name from your editor
      config. After compiling, run `pdfinfo paper_ccis.pdf` (or open Document Properties
      in a PDF reader) and confirm the Author field is empty. Overleaf sometimes injects
      the project owner's name.
- [ ] **Check figure images for identifying content.** `data_pipeline.png` in particular
      may contain absolute file paths (`G:\Projects\...`) or institution names if it was
      exported from a diagram tool.
- [ ] **Decide regular vs short** once you have a page count. 12–16 pp = regular,
      6–8 pp = short. Nothing in between is accepted.
- [ ] **AI disclosure.** SPELLL follows Springer's AI policy. If any part of the writing
      or code was AI-assisted, that must be disclosed — add a statement in the
      camera-ready (not the blind submission, where it could be identifying).
- [ ] **Register a track.** Track 3 (Speech Technologies) is the obvious fit —
      "Speech, voice, and hearing disorders" and "Analysis of speech and audio signals".
      Track 5 lists "Disordered Speech with NLP" as an alternative.

## If the page count lands short of 12

Highest-value additions, in order:
1. A figure of the $h(q)$ curve and $f(\alpha)$ spectrum for one healthy and one
   pathological speaker, side by side. The paper currently formalises the spectrum
   but never shows one.
2. Feature-importance analysis — which of the 12 MFDFA descriptors drive the model.
   You have the trained models; SHAP or permutation importance is cheap to add.
3. A statistical significance test across the five folds (paired t-test or
   Wilcoxon) for the MFDFA-only vs MFDFA+OpenSMILE comparison. The 1.2-point gap
   is currently asserted without a test.

## If it lands over 16

Cut the per-disease section (Table 6) and the two failure-case paragraphs, or
compress Related Work back to unlabelled prose.
