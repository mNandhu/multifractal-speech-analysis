# Multifractal Speech Analysis — Presentation Content

> Slide-ready content for each section. Bullet points are designed to map directly onto PPT slides.

---

## Slide 1 — Title Slide

- **Title:** Multifractal Speech Analysis for Voice Pathology Detection
- **Subtitle:** Automated Voice Disorder Classification Using MFDFA and Acoustic Features
- **Author:** [Your Name]
- **Date:** March 2026

---

## Slide 2 — Introduction & Motivation

- Voice disorders affect millions worldwide — caused by structural, neurological, or functional abnormalities of the vocal folds
- Current clinical diagnosis relies on **laryngoscopic examination** and **perceptual assessment** — both subjective and resource-intensive
- Need for **objective, non-invasive, automated** screening tools
- Speech signals contain rich information about vocal health — but traditional acoustic features (MFCCs, F0) capture only _spectral content_
- **Hypothesis:** The _temporal complexity_ and _scaling dynamics_ of speech — captured by **multifractal analysis** — carry complementary pathology-relevant information

---

## Slide 3 — What is Multifractal Analysis?

- **Fractal signals** exhibit self-similar patterns across time scales
- A **monofractal** signal has one single scaling exponent (Hurst exponent $H$)
- Real speech is **multifractal** — different parts of the signal scale differently
- **MFDFA (Multifractal Detrended Fluctuation Analysis)** quantifies this by computing a _generalised_ Hurst exponent $h(q)$ across statistical moment orders $q$
  - Negative $q$: probes small fluctuations
  - Positive $q$: probes large fluctuations
- If $h(q)$ varies with $q$ → signal is multifractal
- The spread $\Delta h = h(q_{\min}) - h(q_{\max})$ measures the **degree of multifractality**

> _[Insert h(q) vs q plot from `mfdfa_hq_visualization.ipynb`, Cell 14 — the central h(q) comparison between healthy and pathological]_

---

## Slide 4 — Illustrating Multifractality in Speech

- Both healthy and pathological speech are clearly multifractal — $h(q)$ is not constant
- Key differences between healthy and pathological:
  - **Different $\Delta h$ values** — the degree of scaling heterogeneity varies with vocal health
  - **Different singularity spectrum shapes** — width $\Delta\alpha$ and peak position differ
- These differences are what ML models can exploit for classification

> _[Insert waveform comparison from `mfdfa_hq_visualization.ipynb`, Cell 8]_
>
> _[Insert singularity spectrum f(α) plot from `mfdfa_hq_visualization.ipynb`, Cell 16]_

---

## Slide 5 — Problem Definition

- **Primary Task:** Binary classification — **Healthy vs Pathological** voice
- **Secondary Task:** Multi-class classification — identifying the **specific disease group**
  - Neurological (e.g., Recurrent Nerve Paralysis, Spasmodic Dysphonia)
  - Structural (e.g., Vocal Fold Polyp, Reinke's Edema, Laryngitis)
  - Functional (e.g., Hyperfunctional Dysphonia, Psychogenic Dysphonia)
- **Tertiary Task:** Per-disease binary classification — **Disease X vs Healthy**
- **Input:** Sustained vowel /a/ recording at normal pitch (~1–3 seconds)
- **Output:** Predicted class label with confidence

---

## Slide 6 — Gaps Identified

- Most existing voice pathology detection studies use **only spectral/acoustic features** (MFCCs, jitter, shimmer)
- **Temporal complexity** of the speech signal is underexplored as a diagnostic feature
- Many studies have **data leakage** — same speaker's recordings appear in both train and test sets, inflating results
  - We confirmed this: naïve random splitting gave 100% accuracy; proper speaker-grouped splitting dropped it to ~77–89%
- Few studies attempt **fine-grained disease classification** (beyond healthy vs pathological)
- Limited work combining **multifractal features with standard acoustic features** in an ensemble
- Most studies do not address **class imbalance** or **speaker overlap** across pathology categories

---

## Slide 7 — Objectives

1. Extract **multifractal features (MFDFA)** from sustained vowel recordings and evaluate their discriminative power for voice pathology detection
2. Combine MFDFA features with **acoustic (Librosa), OpenSMILE (eGeMAPSv02), and physiological (NeuroKit2)** features
3. Design a **leakage-free evaluation pipeline** with speaker-grouped cross-validation
4. Evaluate classification at multiple granularities:
   - Healthy vs Pathological (binary)
   - Disease group classification (Neurological / Structural / Functional)
   - Per-disease detection
5. Conduct **feature ablation studies** to quantify the contribution of multifractal features

---

## Slide 8 — Dataset Description

- **Source:** Saarbrücken Voice Database (SVD) — publicly available clinical voice database
- **Recording:** Sustained vowels (/a/, /i/, /u/) at multiple pitches + sentences
- **Token used:** Sustained /a/ at normal pitch (`a_n`) — clinical gold standard

| Statistic                     | Value                |
| ----------------------------- | -------------------- |
| Total recordings (all tokens) | 23,119               |
| Unique speakers               | 1,355                |
| Pathology classes             | 12 + healthy         |
| Samples used (token `a_n`)    | 1,656                |
| Sex distribution              | 65% female, 35% male |

---

## Slide 9 — Dataset: Class Distribution

| Pathology                           | Speakers | Samples |
| ----------------------------------- | -------- | ------- |
| Healthy                             | 681      | 687     |
| Hyperfunctional Dysphonia           | 199      | 213     |
| Recurrent Laryngeal Nerve Paralysis | 156      | 213     |
| Laryngitis                          | 128      | 140     |
| Functional Dysphonia                | 108      | 112     |
| Psychogenic Dysphonia               | 80       | 91      |
| Reinke's Edema                      | 54       | 68      |
| Spasmodic Dysphonia                 | 12       | 64      |
| Vocal Fold Polyp                    | 40       | 45      |
| Vocal Fold Nodules                  | 17       | 17      |
| Hypotonic Dysphonia                 | 5        | 5       |
| Parkinson's Disease                 | 1        | 1       |

- Classes with < 50 samples are dropped for grouped experiments
- Healthy class downsampled to match total pathological count

> _[Insert class distribution bar chart from `data_and_features_visualization.ipynb`]_

---

## Slide 10 — Dataset: Disease Grouping

| Group            | Diseases Included                                                                           |
| ---------------- | ------------------------------------------------------------------------------------------- |
| **Neurological** | Parkinson's Disease, Recurrent Laryngeal Nerve Paralysis, Spasmodic Dysphonia               |
| **Structural**   | Vocal Fold Nodules, Vocal Fold Polyp, Reinke's Edema, Laryngitis                            |
| **Functional**   | Hypotonic Dysphonia, Hyperfunctional Dysphonia, Functional Dysphonia, Psychogenic Dysphonia |

- Groups reflect **clinical aetiology** — the underlying cause of the voice disorder
- Reduces the multi-class problem from 12 classes to 3 groups
- Each group has sufficient samples for reliable training

---

## Slide 11 — Methodology: Feature Extraction

**Four feature families extracted per audio sample (~191 total features):**

| Feature Family       | Count | Source     | Description                                                     |
| -------------------- | ----- | ---------- | --------------------------------------------------------------- |
| Acoustic             | 84    | Librosa    | MFCCs, F0, spectral descriptors, energy, ZCR                    |
| Multifractal (MFDFA) | 12    | MFDFA      | $h(q)$ stats, $\tau(q)$ stats, singularity spectrum descriptors |
| OpenSMILE            | 88    | eGeMAPSv02 | Standardised prosodic, spectral, voice quality features         |
| NeuroKit2            | 7     | neurokit2  | Entropy measures, fractal dimensions, signal statistics         |

**MFDFA Parameters:**

- $q$ range: $[-5, 5]$, step $0.5$ (21 values)
- 40 log-spaced window scales
- Polynomial detrending order: 1
- Sample rate: 50 kHz

> _[Insert log-log fluctuation function plot from `mfdfa_hq_visualization.ipynb`, Cell 12]_

---

## Slide 12 — Methodology: Pipeline

```
         ┌─────────────────────┐
         │   SVD Raw Data      │
         │   (.nsp format)     │
         └─────────┬───────────┘
                   ▼
         ┌─────────────────────┐
         │  NSP → WAV Convert  │
         │  (PCM, 50 kHz)      │
         └─────────┬───────────┘
                   ▼
         ┌─────────────────────┐
         │  Dataset Manifest   │
         │  (speaker, pathol.) │
         └─────────┬───────────┘
                   ▼
         ┌─────────────────────┐
         │  Token Filtering    │
         │  (a_n only)         │
         └─────────┬───────────┘
                   ▼
    ┌──────────┬───┴───┬───────────┐
    ▼          ▼       ▼           ▼
┌────────┐ ┌──────┐ ┌────────┐ ┌────────┐
│Acoustic│ │MFDFA │ │OpenSMILE│ │Neurokit│
│(84 ft) │ │(12 ft)│ │(88 ft) │ │(7 ft)  │
└───┬────┘ └──┬───┘ └───┬────┘ └───┬────┘
    └──────────┴─────────┴─────────┘
                   ▼
         ┌─────────────────────┐
         │  Merge + Filter     │
         │  + Collinearity     │
         │  Reduction (r>0.85) │
         └─────────┬───────────┘
                   ▼
         ┌─────────────────────┐
         │  StratifiedGroup    │
         │  KFold (5-fold,     │
         │  speaker-grouped)   │
         └─────────┬───────────┘
                   ▼
    ┌──────────────┴──────────────┐
    ▼                             ▼
┌─────────────────┐   ┌──────────────────┐
│ Binary Model    │   │ Multi-class Model│
│ (Healthy vs     │   │ (Disease Group   │
│  Pathological)  │   │  Classification) │
└─────────────────┘   └──────────────────┘
```

---

## Slide 13 — Methodology: Data Splitting & Leakage Prevention

- **Challenge:** Same speaker may appear in multiple pathology folders; multiple tokens per speaker
- **Solution: Speaker-Grouped Stratified K-Fold**
  - All samples from a speaker in either train OR test — never both
  - Stratified on target label to maintain class proportions
  - 5 folds, `random_seed=42`
- **Additional safeguards:**
  - Single token (`a_n`) per speaker — eliminates within-speaker leakage
  - Mixed-label speakers (healthy + pathological) optionally excluded
  - Overlap speaker audit script for manual verification
- **Class balancing:**
  - Healthy class downsampled (no upsampling)
  - Per-class cap of 500 samples
  - Minimum class threshold (50 samples)

---

## Slide 14 — Methodology: Models

- **Logistic Regression** — linear baseline, class-weight balanced
- **Random Forest** (n=800 trees) — ensemble baseline, balanced subsampling
- **XGBoost** (n=700, lr=0.03) — gradient boosting with regularisation
- **LightGBM** (n=800, lr=0.03) — fast gradient boosting
- **SVM-RBF** (C=3.0, γ=scale) — kernel method, class-weight balanced

**Preprocessing:**

- Tree models: NaN preserved (missing F0 = pathology signal)
- Linear models: Yeo-Johnson transform + imputation
- Collinearity filtering: drop features with $|r| > 0.85$

**Evaluation Metrics:** Accuracy, Balanced Accuracy, Macro F1 (primary)

---

## Slide 15 — Results: Binary Classification (Healthy vs Pathological)

**Best result: 89.1% F1 Macro** (XGBoost, 5-fold speaker-grouped CV)

| Model         | Accuracy  | Balanced Acc. | F1 Macro  |
| ------------- | --------- | ------------- | --------- |
| **XGBoost**   | **0.891** | **0.891**     | **0.891** |
| LightGBM      | 0.885     | 0.885         | 0.885     |
| LogReg        | 0.870     | 0.870         | 0.870     |
| Random Forest | 0.849     | 0.849         | 0.849     |

- All models exceed 84% F1 — healthy vs pathological is a tractable task
- XGBoost and LightGBM consistently outperform linear models

> _[Insert binary confusion matrix from `model_training_v7.ipynb`]_

---

## Slide 16 — Results: Disease Group Classification

**Neurological vs Structural — Best: 64.6% F1 Macro** (Random Forest)

| Model             | Accuracy  | Balanced Acc. | F1 Macro  |
| ----------------- | --------- | ------------- | --------- |
| **Random Forest** | **0.646** | **0.628**     | **0.629** |
| LightGBM          | 0.642     | 0.639         | 0.637     |
| XGBoost           | 0.611     | 0.604         | 0.603     |

- Disease group classification is significantly harder than binary detection
- Structural and neurological groups have distinct but overlapping feature signatures
- Results are substantially above random (50% for 2 classes)

> _[Insert multi-class confusion matrix from `model_training_v7.ipynb`]_

---

## Slide 17 — Results: Per-Disease Detection

| Disease                   | Samples | Best Model    | Balanced Acc. | F1 Macro  |
| ------------------------- | ------- | ------------- | ------------- | --------- |
| Recurrent Nerve Paralysis | 212     | XGBoost       | **86.6%**     | **86.5%** |
| Reinke's Edema            | 67      | LightGBM      | 82.9%         | 82.8%     |
| Laryngitis                | 140     | XGBoost       | 79.6%         | 79.6%     |
| Vocal Fold Polyp          | 44      | LightGBM      | 79.5%         | 79.3%     |
| Functional Dysphonia      | 107     | XGBoost       | 79.4%         | 79.4%     |
| Psychogenic Dysphonia     | 88      | XGBoost       | 79.0%         | 78.8%     |
| Hyperfunctional Dysphonia | 205     | XGBoost       | 77.3%         | 77.3%     |
| Spasmodic Dysphonia       | 64      | LightGBM      | 61.7%         | 58.7%     |
| Vocal Fold Nodules        | 17      | Random Forest | 53.1%         | 48.5%     |

- Neurological conditions (nerve paralysis) are most detectable
- Structural lesions (edema, polyps, laryngitis) show strong detection (79–83%)
- Functional dysphonias show moderate performance (~77–79%)
- Very small sample classes remain challenging

> _[Insert per-disease results bar chart from `model_training_per_disease.ipynb`]_

---

## Slide 18 — Results: Feature Ablation (MFDFA-centric)

| Feature Configuration         | Binary F1 | Multi-class F1 |
| ----------------------------- | --------- | -------------- |
| MFDFA only (12 features)      | 86.6%     | 63.5%          |
| MFDFA + OpenSMILE             | **87.8%** | 62.7%          |
| MFDFA + NeuroKit2             | 86.7%     | 62.3%          |
| MFDFA + NeuroKit2 + OpenSMILE | 87.6%     | **63.6%**      |
| MFDFA + OpenSMILE − Age       | 74.9%     | 61.9%          |

**Key insights:**

- **MFDFA alone achieves 86.6% binary F1** — only 1.2% below the best combination
- OpenSMILE adds modest value (+1.2% F1); NeuroKit2 adds negligible improvement over MFDFA
- **Age is the most impactful single feature** — removing it drops binary F1 by ~13 percentage points
- For disease group classification, MFDFA alone (63.5%) nearly matches the best config (63.6%)

> _[Insert ablation comparison bar chart from `feature_ablation_study.ipynb`]_

---

## Slide 19 — Results: Age Impact Analysis

- Age was consistently the **top feature** in importance rankings
- Removing age from the best config (MFDFA + OpenSMILE):

| Model        | F1 With Age | F1 Without Age | Drop       |
| ------------ | ----------- | -------------- | ---------- |
| LightGBM     | 87.8%       | 74.9%          | **−12.9%** |
| XGBoost      | 87.7%       | 74.4%          | −13.3%     |
| LogReg       | 86.7%       | 72.9%          | −13.8%     |
| RandomForest | 85.8%       | 73.9%          | −11.8%     |

- Age impact on multi-class is much smaller (~1–2% drop)
- **Implication:** Age is a strong predictor for healthy vs pathological, but the models still achieve ~75% F1 using only voice signal features

> _[Insert age impact bar chart from `feature_ablation_study.ipynb`]_

---

## Slide 20 — Analysis: Impact of Proper Evaluation

| Evaluation Strategy           | Binary F1  |
| ----------------------------- | ---------- |
| Random sample split (leakage) | **100%** ⚠ |
| Speaker-grouped, single-fold  | ~75%       |
| Speaker-grouped, 5-fold CV    | **89.1%**  |

- **Data leakage inflates results dramatically** — a common pitfall in voice pathology research
- Proper speaker-level evaluation reveals the true difficulty of the task
- 5-fold grouped CV provides stable, clinically meaningful estimates

---

## Slide 21 — Conclusion

- **Multifractal analysis reveals distinct scaling patterns** in healthy vs pathological speech
  - Confirmed by $h(q)$ visualisation — pathological voices have different multifractal signatures

> _[Insert h(q) vs q plot from `mfdfa_hq_visualization.ipynb`, Cell 14]_

- **89.1% F1 Macro** achieved for healthy vs pathological classification using XGBoost with combined features
- **MFDFA features alone achieve 86.6% F1** — only 1.2% below the best combination, confirming multifractal analysis captures the core discriminative signal
- **Age is the single most impactful feature** (+13% F1), but voice-only features still achieve ~75% F1
- **Speaker-grouped evaluation is critical** — prevents inflated results from data leakage
- Per-disease detection ranges from **48–87% F1** depending on pathology type and available training data
- **Disease group classification (Neurological vs Structural)** achieves ~63% F1 — a challenging but informative task

---

## Slide 22 — Future Work

- **Deep learning:** End-to-end CNN/Transformer models on raw waveforms or spectrograms
- **Hierarchical classification:** Two-stage cascade (screen → diagnose) with calibrated thresholds
- **Multi-token fusion:** Combine information across vowels and pitch conditions systematically
- **Extended complexity features:** Multiscale entropy, permutation entropy, Lyapunov exponents
- **Clinical validation:** Prospective study on unseen clinical cohorts
- **Explainability:** SHAP-based analysis to identify which multifractal features drive individual predictions

---

## Appendix Slides (if needed)

### MFDFA Parameters

| Parameter            | Value                 |
| -------------------- | --------------------- |
| $q$ range            | $[-5, 5]$, step $0.5$ |
| Number of $q$ values | 21                    |
| Number of scales     | 40 (log-spaced)       |
| Polynomial order     | 1 (linear detrending) |
| Sample rate          | 50,000 Hz             |

### Models & Hyperparameters

| Model               | Key Parameters                                           |
| ------------------- | -------------------------------------------------------- |
| Logistic Regression | max_iter=3000, C=1.0, balanced                           |
| Random Forest       | n_estimators=800, min_samples_leaf=2, balanced_subsample |
| XGBoost             | n_estimators=700, lr=0.03, max_depth=6, subsample=0.8    |
| LightGBM            | n_estimators=800, lr=0.03, num_leaves=63, subsample=0.8  |
| SVM-RBF             | C=3.0, γ=scale, balanced                                 |

### Summary Table from MFDFA Visualisation

> _[Insert summary table from `mfdfa_hq_visualization.ipynb`, Cell 18]_
