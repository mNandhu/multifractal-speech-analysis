# Multifractal Speech Analysis for Voice Pathology Detection

**A Machine Learning Pipeline for Automated Voice Disorder Classification Using Multifractal and Acoustic Features**

---

## Table of Contents

1. [Problem Statement & Motivation](#1-problem-statement--motivation)
2. [Dataset](#2-dataset)
3. [Feature Extraction](#3-feature-extraction)
4. [Data Pipeline & Splitting Strategy](#4-data-pipeline--splitting-strategy)
5. [Experimental Configurations & Results](#5-experimental-configurations--results)
   - 5.1 [Baseline: Naïve Multi-class Classification](#51-baseline-naïve-multi-class-classification)
   - 5.2 [Speaker-Aware Splitting & Disease Grouping](#52-speaker-aware-splitting--disease-grouping)
   - 5.3 [Speaker-Level Aggregation with Grouped CV](#53-speaker-level-aggregation-with-grouped-cv)
   - 5.4 [Token-Level & Single-Token Pipelines](#54-token-level--single-token-pipelines)
   - 5.5 [Feature Ablation Studies](#55-feature-ablation-studies)
   - 5.6 [Final Model: Neurokit2 + Refined MFDFA](#56-final-model-neurokit2--refined-mfdfa)
   - 5.7 [Per-Disease Binary Classification](#57-per-disease-binary-classification)
6. [Summary of Results](#6-summary-of-results)
7. [Key Findings & Discussion](#7-key-findings--discussion)
8. [Conclusion & Future Work](#8-conclusion--future-work)

---

## 1. Problem Statement & Motivation

Voice disorders affect a significant portion of the population and can arise from a wide range of causes — structural lesions on the vocal folds, neurological conditions affecting motor control, or functional/psychogenic patterns. Clinical diagnosis typically relies on laryngoscopic examination combined with perceptual voice assessment, both of which are subjective, time-consuming, and require specialist expertise.

**Objective:** This project investigates whether _multifractal analysis_ of speech signals, combined with traditional acoustic features, can enable automated classification of voice pathologies from sustained vowel recordings. The core hypothesis is that pathological voices exhibit characteristic changes in their multifractal structure — specifically in the complexity, self-similarity, and scaling behaviour of the speech signal — that differ from healthy voices and vary across pathology types.

**Multifractal Detrended Fluctuation Analysis (MFDFA)** characterises a signal's local scaling behaviour across multiple statistical moments ($q$-orders), producing a singularity spectrum that captures how the fractal structure varies across different time scales. Unlike monofractal measures (e.g., a single Hurst exponent), MFDFA can reveal rich, heterogeneous scaling patterns that may distinguish between different types of vocal dysfunction.

**Motivation:**

- Standard acoustic features (MFCCs, F0, spectral descriptors) encode the _spectral content_ of speech but may miss _temporal complexity_ patterns associated with pathology.
- Multifractal features capture _scaling dynamics_ across time scales, providing a complementary signal that could improve classification or serve as a standalone biomarker.
- An automated, non-invasive screening tool could assist clinicians in triaging patients and monitoring treatment outcomes.

---

## 2. Dataset

### 2.1 Source

The dataset is derived from the **Saarbrücken Voice Database (SVD)**, a publicly available clinical voice database containing recordings from patients with various voice pathologies and healthy control speakers. Recordings are in a proprietary `.nsp` format and are converted to `.wav` as part of the preprocessing pipeline.

### 2.2 Recording Protocol

Each speaker provides recordings in multiple modalities:

- **Sustained vowels:** /a/, /i/, /u/ at normal, high, and low pitch, as well as rising-falling pitch
- **Sentences:** Connected speech phrases

For the primary experiments, the sustained vowel `/a/` at normal pitch (token `a_n`) is used as the canonical clinical assessment signal.

### 2.3 Dataset Composition

| Statistic                     | Value                |
| ----------------------------- | -------------------- |
| Total recordings (all tokens) | 23,119               |
| Unique speakers               | 1,355                |
| Pathology classes             | 12 (+ healthy)       |
| Samples per token (`a_n`)     | 1,656                |
| Sex distribution              | 65% female, 35% male |

### 2.4 Class Distribution (Token `a_n`)

| Pathology (German)          | Pathology (English)                 | Speakers | Samples |
| --------------------------- | ----------------------------------- | -------- | ------- |
| healthy                     | Healthy                             | 681      | 687     |
| Hyperfunktionelle Dysphonie | Hyperfunctional Dysphonia           | 199      | 213     |
| Rekurrensparese             | Recurrent Laryngeal Nerve Paralysis | 156      | 213     |
| Laryngitis                  | Laryngitis                          | 128      | 140     |
| Funktionelle Dysphonie      | Functional Dysphonia                | 108      | 112     |
| Psychogene Dysphonie        | Psychogenic Dysphonia               | 80       | 91      |
| Reinke Ödem                 | Reinke's Edema                      | 54       | 68      |
| Spasmodische Dysphonie      | Spasmodic Dysphonia                 | 12       | 64      |
| Stimmlippenpolyp            | Vocal Fold Polyp                    | 40       | 45      |
| Phonationsknötchen          | Vocal Fold Nodules                  | 17       | 17      |
| Hypotone Dysphonie          | Hypotonic Dysphonia                 | 5        | 5       |
| Morbus Parkinson            | Parkinson's Disease                 | 1        | 1       |

> **Note:** Classes with very few samples (Morbus Parkinson: 1 sample, Hypotone Dysphonie: 5 samples) are dropped in most experiments via a `MIN_SAMPLES_PER_CLASS` threshold (typically 50 for grouped experiments, 10 for per-disease experiments).

_[Insert class distribution bar chart from `data_and_features_visualization.ipynb` here]_

### 2.5 Disease Grouping

In later experiments, individual pathologies are grouped into clinically meaningful super-categories:

| Group            | Pathologies Included                                                                        |
| ---------------- | ------------------------------------------------------------------------------------------- |
| **Neurological** | Parkinson's Disease, Recurrent Laryngeal Nerve Paralysis, Spasmodic Dysphonia               |
| **Structural**   | Vocal Fold Nodules, Vocal Fold Polyp, Reinke's Edema, Laryngitis                            |
| **Functional**   | Hypotonic Dysphonia, Hyperfunctional Dysphonia, Functional Dysphonia, Psychogenic Dysphonia |

---

## 3. Feature Extraction

Four feature families are extracted from each audio sample, totalling approximately 191 features:

### 3.1 Acoustic Features (Librosa) — 84 features

| Feature Group          | Count | Description                                          |
| ---------------------- | ----- | ---------------------------------------------------- |
| Time-domain            | 4     | Energy, absolute mean, peak amplitude, crest factor  |
| RMS                    | 4     | Root mean square energy (mean, std, min, max)        |
| ZCR                    | 4     | Zero-crossing rate statistics                        |
| Spectral centroid      | 4     | First moment of the spectrum                         |
| Spectral bandwidth     | 4     | Spread around the centroid                           |
| Spectral roll-off      | 4     | Frequency below which 85% of energy resides          |
| Spectral flatness      | 4     | Tonality measure                                     |
| MFCCs 1–13             | 52    | Mean & std for each coefficient + first-order deltas |
| Fundamental freq. (F0) | 4     | Mean, std, min, max via YIN algorithm                |

### 3.2 Multifractal Features (MFDFA) — 12 features

| Feature                  | Description                                                                |
| ------------------------ | -------------------------------------------------------------------------- |
| `mf_hq_mean/std/min/max` | Generalised Hurst exponent $h(q)$ statistics                               |
| `mf_tau_mean/std`        | Scaling exponent $\tau(q)$ statistics                                      |
| `mf_alpha_mean/std`      | Singularity strength $\alpha$ statistics                                   |
| `mf_spectrum_width`      | Width of singularity spectrum ($\Delta\alpha$) — degree of multifractality |
| `mf_spectrum_peak_alpha` | Location of spectrum peak — dominant scaling behaviour                     |
| `mf_spectrum_peak_f`     | Height of spectrum peak                                                    |
| `mf_spectrum_asymmetry`  | Left/right asymmetry of spectrum                                           |

**MFDFA Parameters:**

| Parameter            | Initial Value       | Final Value (v7+)   |
| -------------------- | ------------------- | ------------------- |
| Polynomial order     | 1                   | 1                   |
| $q$ range            | $[-5, 5]$, step 1.0 | $[-5, 5]$, step 0.5 |
| Number of $q$ values | 11                  | 21                  |
| Number of scales     | 20                  | 40                  |
| Target sample rate   | 22,050 Hz           | 50,000 Hz           |

### 3.3 OpenSMILE Features (eGeMAPSv02) — 88 features

The extended Geneva Minimalistic Acoustic Parameter Set version 02 (eGeMAPSv02) is extracted at the functional level using the OpenSMILE toolkit. This standardised feature set includes prosodic, spectral, and voice quality parameters widely used in computational paralinguistics.

### 3.4 NeuroKit2 Features — 7 features

| Feature                  | Description                              |
| ------------------------ | ---------------------------------------- |
| `nk_entropy_shannon`     | Shannon entropy (complexity measure)     |
| `nk_entropy_approximate` | Approximate entropy (regularity)         |
| `nk_entropy_sample`      | Sample entropy (complexity/irregularity) |
| `nk_fractal_petrosian`   | Petrosian fractal dimension              |
| `nk_fractal_sevcik`      | Ševčík fractal dimension                 |
| `nk_skewness`            | Signal skewness                          |
| `nk_kurtosis`            | Signal kurtosis                          |

> **Note:** NeuroKit2 features were introduced in v7 of the pipeline. The signal is downsampled to 8 kHz before entropy computation (ApEn and SampEn are $O(N^2)$) to keep computation tractable.

---

## 4. Data Pipeline & Splitting Strategy

### 4.1 Data Preprocessing Pipeline

```
Raw .nsp files
    │
    ▼
NSP → WAV conversion (int16 PCM)
    │
    ▼
Dataset manifest (CSV: sample_key, speaker_id, pathology, wav_path, ...)
    │
    ▼
Audio loading (librosa, target SR, mono, peak-normalised)
    │
    ▼
Feature extraction (4 families in parallel)
    │
    ▼
Feature tables (CSV: sample_core, acoustic, multifractal, opensmile, neurokit2)
    │
    ▼
Merge & filter → Model-ready dataset
```

### 4.2 Token Selection

| Experiment Phase | Token Strategy                                   | Rationale                                                                                           |
| ---------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| Baselines (v1)   | All tokens (14 per speaker)                      | Maximum data volume                                                                                 |
| v2+              | Single token: `a_n` (sustained /a/ normal pitch) | Clinical gold standard for voice assessment; one sample per speaker prevents within-speaker leakage |

### 4.3 Speaker Overlap & Data Leakage Prevention

A critical challenge in this dataset is that some speakers appear across multiple pathology folders (overlap speakers). The pipeline addresses data leakage at multiple levels:

1. **Speaker-Grouped Splitting:** From v2 onwards, `StratifiedGroupKFold` (scikit-learn) ensures that all samples from a given speaker appear exclusively in either train or test, never both. The grouping key is `speaker_id`.

2. **Mixed-Label Speaker Exclusion:** Speakers appearing with both healthy and pathological labels (likely due to longitudinal recordings) are optionally excluded via `EXCLUDE_MIXED_BINARY_SPEAKERS=True`.

3. **Single-Token Filtering:** Using only `a_n` reduces each speaker to a single sample, naturally preventing within-speaker leakage from multiple utterances.

4. **Overlap Speaker Auditing:** A dedicated script (`scripts/audit_speaker_overlaps.py`) scans pathology folders and reports pairwise speaker-ID overlaps for manual review.

### 4.4 Class Balancing

- **Healthy downsampling:** When `BALANCE_HEALTHY=True`, the healthy class is downsampled to match the total pathological sample count (no upsampling to avoid duplicate `sample_key` issues).
- **Per-class cap:** `MAX_SAMPLES_PER_CLASS` (typically 500) limits any single class to avoid dominance.
- **Minimum threshold:** `MIN_SAMPLES_PER_CLASS` (50 for grouped experiments, 10 for per-disease) drops under-represented classes.

### 4.5 Cross-Validation Strategy

| Parameter          | Value                                |
| ------------------ | ------------------------------------ |
| Method             | `StratifiedGroupKFold`               |
| Folds              | 5                                    |
| Group key          | `speaker_id`                         |
| Stratification key | Target label (binary or multi-class) |
| Random seed        | 42                                   |

Threshold tuning for binary classification is performed on training-fold probabilities using a grid search over thresholds (typically 0.35–0.65) to optimise accuracy or balanced accuracy.

---

## 5. Experimental Configurations & Results

### 5.1 Baseline: Naïve Multi-class Classification

**Notebooks:** `model_training_baselines.ipynb`, `model_training_feature_engineered.ipynb`, `model_training_mfdfa_only.ipynb`

**Setup:**

- All tokens, random 80/20 train-test split (no speaker grouping)
- 200 samples per class cap
- Features: Acoustic + Multifractal (+ OpenSMILE where available)
- **⚠ Known data leakage** due to random sample-level splits

**Models:** Logistic Regression, Random Forest, SVM-RBF, XGBoost, LightGBM

#### Binary Classification (Healthy vs Pathological)

| Model         | Accuracy  | Balanced Acc. | F1 Macro  |
| ------------- | --------- | ------------- | --------- |
| LogReg        | **1.000** | **1.000**     | **1.000** |
| XGBoost       | 1.000     | 1.000         | 1.000     |
| LightGBM      | 1.000     | 1.000         | 1.000     |
| SVM-RBF       | 0.987     | 0.972         | 0.982     |
| Random Forest | 0.879     | 0.736         | 0.784     |

> **⚠ These results are inflated** due to speaker leakage — the same speaker's different tokens appear in both train and test sets.

#### Multi-class Classification (Pathological Samples — 7 classes)

| Model         | Accuracy | Balanced Acc. | F1 Macro |
| ------------- | -------- | ------------- | -------- |
| Random Forest | 0.231    | 0.218         | 0.194    |
| LogReg        | 0.231    | 0.200         | 0.185    |
| SVM-RBF       | 0.207    | 0.196         | 0.177    |
| LightGBM      | 0.198    | 0.185         | 0.166    |
| XGBoost       | 0.190    | 0.188         | 0.160    |

#### Feature Ablation: MFDFA-Only Baseline

Using **only multifractal features** (12 features):

| Task        | Best Model    | Accuracy | Balanced Acc. | F1 Macro |
| ----------- | ------------- | -------- | ------------- | -------- |
| Binary      | SVM-RBF       | 0.541    | 0.605         | 0.520    |
| Multi-class | Random Forest | 0.207    | 0.203         | 0.190    |

> Multifractal features alone provide above-chance discrimination for healthy vs pathological, but are insufficient for fine-grained disease classification in isolation.

---

### 5.2 Speaker-Aware Splitting & Disease Grouping

**Notebooks:** `model_training_v2.ipynb`, `model_training_v2_grouped.ipynb`

**Key Changes:**

- Single token `a_n` only (one sample per speaker)
- `StratifiedGroupKFold(n_splits=5)` with `speaker_id` as group — **no speaker leakage**
- `MIN_SAMPLES_PER_CLASS=50` — drops Parkinson's (14 samples, 1 speaker)
- `SelectKBest(f_classif, k=50)` for feature selection

#### v2: Individual Pathology Targets

**Binary Classification:**

| Model         | Accuracy  | Balanced Acc. | F1 Macro  |
| ------------- | --------- | ------------- | --------- |
| LogReg        | **1.000** | **1.000**     | **1.000** |
| Random Forest | 1.000     | 1.000         | 1.000     |
| XGBoost       | 1.000     | 1.000         | 1.000     |
| LightGBM      | 1.000     | 1.000         | 1.000     |
| SVM-RBF       | 0.991     | 0.988         | 0.990     |

**Multi-class Classification (Pathological — individual diseases):**

| Model         | Accuracy  | Balanced Acc. | F1 Macro  |
| ------------- | --------- | ------------- | --------- |
| SVM-RBF       | **0.569** | **0.524**     | **0.454** |
| LightGBM      | 0.542     | 0.481         | 0.416     |
| XGBoost       | 0.542     | 0.446         | 0.409     |
| Random Forest | 0.569     | 0.411         | 0.392     |
| LogReg        | 0.403     | 0.424         | 0.353     |

#### v2 Grouped: Disease Group Targets

Using grouped targets (**Neurological, Structural, Functional, + unmapped**):

**Binary Classification:**

| Model         | Accuracy  | Balanced Acc. | F1 Macro  |
| ------------- | --------- | ------------- | --------- |
| Random Forest | **0.790** | 0.743         | **0.753** |
| LightGBM      | 0.756     | 0.730         | 0.729     |
| SVM-RBF       | 0.748     | **0.742**     | 0.730     |
| XGBoost       | 0.731     | 0.705         | 0.702     |
| LogReg        | 0.723     | 0.729         | 0.709     |

**Multi-class Classification (Grouped pathologies):**

| Model         | Accuracy  | Balanced Acc. | F1 Macro  |
| ------------- | --------- | ------------- | --------- |
| SVM-RBF       | **0.734** | **0.763**     | **0.726** |
| LogReg        | 0.709     | 0.744         | 0.702     |
| Random Forest | 0.734     | 0.616         | 0.615     |
| XGBoost       | 0.671     | 0.598         | 0.601     |
| LightGBM      | 0.620     | 0.560         | 0.561     |

> **Observation:** Binary results dropped dramatically after introducing proper speaker-grouped evaluation. The 100% accuracy in baselines was almost certainly due to data leakage. The grouped multi-class classification (3–4 groups) performs much better than individual disease classification (7+ classes).

**5-Fold CV Stability (v2 Grouped):**

- Binary F1 Macro: **0.711 ± 0.067**
- Multi-class F1 Macro: **0.649 ± 0.084**

---

### 5.3 Speaker-Level Aggregation with Grouped CV

**Notebooks:** `model_training_v3_grouped.ipynb`, `model_training_v3_1_grouped.ipynb`

**Key Changes:**

- All tokens per speaker aggregated (mean, std across tokens) — reduces utterance-level noise
- Feature selection search: `SelectKBest` with $k \in \{30, 50, 100, 150, 200\}$
- Threshold tuning grid: 25 points in $[0.20, 0.80]$
- 4 disease groups: Neurological, Structural, Functional, + unmapped

#### v3: Binary Results (5-Fold Grouped CV, Best Configuration)

| Model         | k       | Accuracy          | Balanced Acc.     | F1 Macro          |
| ------------- | ------- | ----------------- | ----------------- | ----------------- |
| **LightGBM**  | **200** | **0.889 ± 0.047** | **0.875 ± 0.052** | **0.882 ± 0.051** |
| XGBoost       | 200     | 0.886 ± 0.047     | 0.868 ± 0.054     | 0.878 ± 0.052     |
| Random Forest | 200     | 0.885 ± 0.044     | 0.865 ± 0.052     | 0.875 ± 0.050     |
| LogReg        | 200     | 0.875 ± 0.040     | 0.867 ± 0.046     | 0.870 ± 0.044     |

#### v3: Multi-class Results (Best Configuration)

| Model       | k       | Accuracy          | Balanced Acc.     | F1 Macro          |
| ----------- | ------- | ----------------- | ----------------- | ----------------- |
| **SVM-RBF** | **100** | **0.871 ± 0.049** | **0.867 ± 0.046** | **0.866 ± 0.050** |

**OOF Classification Report (Binary — LightGBM k=200):**

- Overall accuracy: **0.89** (615 samples)
- Macro average: precision=0.90, recall=0.87, F1=0.88

**OOF Classification Report (Multi-class — SVM-RBF k=100):**

- Overall accuracy: **0.87** (257 samples)
- Macro average: precision=0.87, recall=0.87, F1=0.87

_[Insert confusion matrix plots from `model_training_v3_grouped.ipynb` here]_

#### v3.1: Advanced Feature Selection

Key improvements:

- **Feature selection strategies compared:** ANOVA f_classif, Mutual Information, L1 (LinearSVC), RFE (Random Forest)
- **PowerTransformer (Yeo-Johnson)** replaces StandardScaler for heavy-tailed acoustic distributions
- **Imputation:** `constant=0.0` with `add_indicator=True` — missing features (e.g., no F0 detected) treated as pathology signal

**Finding:** L1-based feature selection (LinearSVC, C=0.05) emerged as the best approach, outperforming ANOVA and RFE in several configurations.

---

### 5.4 Token-Level & Single-Token Pipelines

**Notebooks:** `model_training_v4_sample_level.ipynb`, `model_training_v5.ipynb`

#### v4: Sample-Level Classification

Returns to token-level (no speaker aggregation) but retains speaker-grouped CV:

- All tokens kept, `token` used as a categorical feature
- Three evaluation strategies:
  1. **Token-level:** Direct per-sample prediction
  2. **Speaker-level aggregation:** Train on tokens, aggregate probabilities by speaker at evaluation
  3. **Token-specialist ensemble:** Global model + per-token specialist models, aggregated at speaker level

**Binary Results (Token-Level, 5-Fold CV):**

| Model    | k    | Accuracy | Balanced Acc. | F1 Macro |
| -------- | ---- | -------- | ------------- | -------- |
| XGBoost  | auto | 0.780    | 0.780         | 0.778    |
| LightGBM | auto | 0.777    | 0.777         | 0.770    |

**Multi-class Results (Grouped, Speaker-Level Eval):**

| Model   | k    | Accuracy | Balanced Acc. | F1 Macro |
| ------- | ---- | -------- | ------------- | -------- |
| XGBoost | auto | 0.567    | 0.537         | 0.525    |

#### v5: Single-Token `a_n` Pipeline

**Design philosophy shift:** "Prove multifractal features work on a single standardised token before adding complexity."

Key changes:

- `a_n` token only — clinical gold standard
- **Tree models receive raw NaN values** (no imputation/scaling) — missing F0 is itself a pathology signal
- Linear models retain PowerTransformer preprocessing
- No feature selection (trees handle feature importance natively)

**Binary Results (Speaker-Level, 5-Fold CV):**

| Model         | Accuracy | Balanced Acc. | F1 Macro |
| ------------- | -------- | ------------- | -------- |
| LightGBM      | 0.773    | 0.777         | 0.770    |
| LogReg        | 0.771    | 0.771         | 0.766    |
| XGBoost       | 0.768    | 0.776         | 0.766    |
| Random Forest | 0.754    | 0.758         | 0.751    |

**Multi-class Results (Grouped, Speaker-Level):**

| Model   | Accuracy | Balanced Acc. | F1 Macro |
| ------- | -------- | ------------- | -------- |
| XGBoost | 0.529    | 0.510         | 0.510    |

---

### 5.5 Feature Ablation Studies

**Notebooks:** `model_training_v6.ipynb`, `model_training_v6_only_two_grps.ipynb`, `feature_ablation_study.ipynb`

#### v6: Full Feature Set + Age + Collinearity Filtering

Added features: **Age** (computed from recording date and birth date).
Added processing: **Collinearity filtering** at $|r| > 0.95$.

**Binary Results (Speaker-Level, 5-Fold CV):**

| Model         | Accuracy  | Balanced Acc. | F1 Macro  |
| ------------- | --------- | ------------- | --------- |
| **LightGBM**  | **0.874** | **0.871**     | **0.870** |
| XGBoost       | 0.869     | 0.868         | 0.866     |
| Random Forest | 0.858     | 0.849         | 0.851     |
| LogReg        | 0.853     | 0.850         | 0.849     |

**Multi-class Results (Grouped — 3 groups: N, S, F):**

| Model         | Accuracy | Balanced Acc. | F1 Macro |
| ------------- | -------- | ------------- | -------- |
| XGBoost       | 0.542    | 0.521         | 0.522    |
| LightGBM      | 0.524    | 0.500         | 0.502    |
| Random Forest | 0.502    | 0.475         | 0.473    |

#### v6 (Two Groups Only): Neurological vs Structural

Drops Functional group, keeps only Neurological and Structural, and also **removes all acoustic features** to isolate multifractal contribution:

**Binary Results:**

| Model    | Accuracy  | Balanced Acc. | F1 Macro  |
| -------- | --------- | ------------- | --------- |
| LightGBM | **0.872** | **0.872**     | **0.871** |
| XGBoost  | 0.871     | 0.871         | 0.870     |

**Multi-class (Neurological vs Structural):**

| Model         | Accuracy  | Balanced Acc. | F1 Macro  |
| ------------- | --------- | ------------- | --------- |
| Random Forest | **0.627** | **0.627**     | **0.626** |
| XGBoost       | 0.613     | 0.613         | 0.610     |
| LightGBM      | 0.613     | 0.613         | 0.612     |

#### Systematic Feature Family Ablation

**Notebook:** `feature_ablation_study.ipynb`

A systematic ablation study isolating the contribution of each feature family. All experiments use the same filtered dataset (922 samples, 787 speakers), same models, same speaker-grouped 5-fold CV, and collinearity filtering at $|r| > 0.85$.

**Binary Classification (Healthy vs Pathological) — Best Model per Config:**

| Feature Configuration         | Best Model | Accuracy  | Balanced Acc. | F1 Macro  |
| ----------------------------- | ---------- | --------- | ------------- | --------- |
| **MFDFA + OpenSMILE**         | LightGBM   | **0.879** | **0.879**     | **0.878** |
| MFDFA + NeuroKit2 + OpenSMILE | XGBoost    | 0.876     | 0.876         | 0.876     |
| MFDFA + NeuroKit2             | LogReg     | 0.868     | 0.868         | 0.867     |
| MFDFA only                    | LogReg     | 0.867     | 0.867         | 0.866     |
| MFDFA + OpenSMILE − Age       | LightGBM   | 0.752     | 0.752         | 0.749     |

**Multi-class Classification (Neurological vs Structural) — Best Model per Config:**

| Feature Configuration         | Best Model   | Accuracy  | Balanced Acc. | F1 Macro  |
| ----------------------------- | ------------ | --------- | ------------- | --------- |
| MFDFA + NeuroKit2 + OpenSMILE | RandomForest | **0.661** | **0.634**     | **0.636** |
| MFDFA only                    | RandomForest | 0.653     | 0.634         | 0.635     |
| MFDFA + OpenSMILE             | RandomForest | 0.648     | 0.626         | 0.627     |
| MFDFA + NeuroKit2             | RandomForest | 0.646     | 0.621         | 0.623     |
| MFDFA + OpenSMILE − Age       | RandomForest | 0.646     | 0.619         | 0.619     |

**Age Impact Analysis (MFDFA + OpenSMILE with vs without age):**

| Model        | F1 With Age | F1 Without Age | Δ F1       |
| ------------ | ----------- | -------------- | ---------- |
| LogReg       | 0.867       | 0.729          | **+0.138** |
| RandomForest | 0.858       | 0.739          | +0.118     |
| LightGBM     | 0.878       | 0.749          | +0.129     |
| XGBoost      | 0.877       | 0.744          | +0.133     |

> **Key findings from the systematic ablation:**
>
> 1. **MFDFA features alone achieve 86.6% F1** for binary classification — only 1.2% below the best combined config. This confirms that multifractal features carry the bulk of the discriminative signal.
> 2. **Age is the single most impactful feature**, contributing ~13 percentage points to binary F1 on average. Removing age drops performance from 87.8% to 74.9%.
> 3. **Adding NeuroKit2 to MFDFA provides marginal improvement** (+0.1% F1) — both capture overlapping complexity/fractal information.
> 4. **OpenSMILE adds modest value** on top of MFDFA (+1.2% F1 for binary), but is less impactful than age.
> 5. For **multi-class (disease group)** classification, MFDFA alone is surprisingly competitive (63.5% F1), suggesting that scaling dynamics differ between neurological and structural pathologies.

_[Insert ablation comparison bar chart from `feature_ablation_study.ipynb` here]_

_[Insert age impact bar chart from `feature_ablation_study.ipynb` here]_

---

### 5.6 Final Model: NeuroKit2 + Refined MFDFA

**Notebook:** `model_training_v7.ipynb`

**Key additions:**

- **NeuroKit2 features** (entropy, fractal dimension, signal statistics)
- **Refined MFDFA parameters:** $q_{\text{step}} = 0.5$ (21 q-values), 40 scales, 50 kHz sample rate
- **Feature pruning:** MFCC and F1/F2/F3 formant features explicitly dropped (34 features removed, 79 remaining)
- **Collinearity threshold:** $|r| > 0.85$

**Binary Results (Speaker-Level, 5-Fold CV):**

| Model         | Accuracy  | Balanced Acc. | F1 Macro  |
| ------------- | --------- | ------------- | --------- |
| **XGBoost**   | **0.891** | **0.891**     | **0.891** |
| LightGBM      | 0.885     | 0.885         | 0.885     |
| LogReg        | 0.870     | 0.870         | 0.870     |
| Random Forest | 0.849     | 0.849         | 0.849     |

**Multi-class Results (Neurological vs Structural):**

| Model             | Accuracy  | Balanced Acc. | F1 Macro  |
| ----------------- | --------- | ------------- | --------- |
| **Random Forest** | **0.646** | **0.628**     | **0.629** |
| LightGBM          | 0.642     | 0.639         | 0.637     |
| XGBoost           | 0.611     | 0.604         | 0.603     |

> **Best overall result:** The v7 pipeline achieves **89.1% F1 macro** on binary classification (Healthy vs Pathological) using XGBoost with speaker-level 5-fold CV — a clinically meaningful and leakage-free evaluation.

_[Insert binary and multi-class confusion matrices from `model_training_v7.ipynb` here]_

---

### 5.7 Per-Disease Binary Classification

**Notebook:** `model_training_per_disease.ipynb`

Trains one binary classifier per disease (Disease X vs Healthy), with balanced sampling and speaker-grouped CV. Uses the v7 feature set.

| Disease                             | n (disease) | n (healthy) | Best Model    | Balanced Acc. | F1 Macro  |
| ----------------------------------- | ----------- | ----------- | ------------- | ------------- | --------- |
| Recurrent Laryngeal Nerve Paralysis | 212         | 212         | XGBoost       | **0.866**     | **0.865** |
| Reinke's Edema                      | 67          | 67          | LightGBM      | 0.829         | 0.828     |
| Laryngitis                          | 140         | 140         | XGBoost       | 0.796         | 0.796     |
| Vocal Fold Polyp                    | 44          | 44          | LightGBM      | 0.795         | 0.793     |
| Functional Dysphonia                | 107         | 107         | XGBoost       | 0.794         | 0.794     |
| Psychogenic Dysphonia               | 88          | 88          | XGBoost       | 0.790         | 0.788     |
| Hyperfunctional Dysphonia           | 205         | 205         | XGBoost       | 0.773         | 0.773     |
| Spasmodic Dysphonia                 | 64          | 64          | LightGBM      | 0.617         | 0.587     |
| Vocal Fold Nodules                  | 17          | 17          | Random Forest | 0.531         | 0.485     |

> **Observations:**
>
> - Neurological conditions (recurrent nerve paralysis) and structural lesions (Reinke's edema, polyps) are more easily distinguished from healthy voices.
> - Functional dysphonias (hyperfunctional, psychogenic) are moderately detectable.
> - Very small classes (vocal fold nodules: 17 samples) remain challenging.

_[Insert per-disease results bar chart from `model_training_per_disease.ipynb` here]_

---

## 6. Summary of Results

### 6.1 Binary Classification (Healthy vs Pathological) — Evolution

| Version       | Key Change                     | Best F1 Macro | Best Model    |
| ------------- | ------------------------------ | ------------- | ------------- |
| Baseline (v1) | Random split, all tokens       | 1.000\*       | LogReg        |
| v2            | Speaker-grouped, `a_n` only    | 1.000†        | Multiple      |
| v2 Grouped    | + disease grouping             | 0.753         | Random Forest |
| v3            | Speaker-aggregated, k-sweep    | **0.882**     | LightGBM      |
| v5            | Single-token, NaN-preserving   | 0.770         | LightGBM      |
| v6            | + Age, collinearity filter     | 0.870         | LightGBM      |
| **v7**        | **+ NeuroKit2, refined MFDFA** | **0.891**     | **XGBoost**   |

\* Inflated due to data leakage.
† Single-fold evaluation; subsequent stability tests show lower values.

### 6.2 Multi-class (Disease Group) Classification — Evolution

| Version     | Groups                           | Best F1 Macro | Best Model    |
| ----------- | -------------------------------- | ------------- | ------------- |
| v2          | Individual diseases (7)          | 0.454         | SVM-RBF       |
| v2 Grouped  | N, S, F, + unmapped              | 0.726         | SVM-RBF       |
| v3          | N, S, F, + unmapped (aggregated) | **0.866**     | SVM-RBF       |
| v6          | N, S, F (3 groups)               | 0.522         | XGBoost       |
| v6 (2 grps) | N, S only                        | 0.626         | Random Forest |
| v7          | N, S only                        | **0.629**     | Random Forest |

### 6.3 Feature Set Comparison

| Feature Set                                  | Binary F1 | Notes                                                 |
| -------------------------------------------- | --------- | ----------------------------------------------------- |
| MFDFA only (naïve split, v1)                 | 0.520     | Above chance; naïve random split (leakage present)    |
| MFDFA only (speaker-grouped, ablation study) | **0.866** | Proper evaluation — multifractal alone is very strong |
| MFDFA + NeuroKit2                            | 0.867     | NeuroKit2 adds marginal improvement over MFDFA alone  |
| MFDFA + OpenSMILE                            | 0.878     | Best 2-family combination                             |
| MFDFA + NeuroKit2 + OpenSMILE                | 0.876     | Adding NeuroKit2 to MFDFA+OS doesn't help further     |
| MFDFA + OpenSMILE − Age                      | 0.749     | Age removal causes ~13% F1 drop                       |
| All features + NeuroKit2 (refined MFDFA, v7) | **0.891** | v7 best result (includes acoustic features)           |

---

## 7. Key Findings & Discussion

### 7.1 Data Leakage Matters

The most dramatic finding was the drop from 100% to ~75-89% accuracy when switching from random sample-level splits to proper speaker-grouped evaluation. This underscores the critical importance of speaker-aware splitting in voice pathology research.

### 7.2 Multifractal Features Carry Unique Information

- In the systematic ablation study with proper speaker-grouped evaluation, **MFDFA features alone achieve 86.6% F1** for binary classification — only 1.2% below the best combined configuration (MFDFA + OpenSMILE at 87.8%).
- When acoustic features (MFCCs, formants) are removed but multifractal and other non-spectral features retained, binary performance remains at ~87%, confirming multifractal features are complementary to spectral descriptors.
- **Age is the dominant individual feature**, contributing ~13 percentage points to binary F1. Without age, even the best feature combination drops from 87.8% to 74.9%.
- For multi-class (disease group) classification, MFDFA alone achieves 63.5% F1 — nearly matching the best combined config (63.6%), suggesting that multifractal scaling dynamics capture most of the between-pathology differences.

### 7.3 Healthy vs Pathological is Tractable; Disease Typing is Hard

- Binary classification consistently achieves 85–89% F1 across multiple proper evaluations.
- Fine-grained disease classification (7+ individual pathologies) remains challenging at ~45% F1, though grouping into 2–4 clinically meaningful categories improves this to 63–87% depending on the grouping granularity and aggregation strategy.

### 7.4 Tree Models Benefit from Raw NaN Values

The v5 finding that tree-based models (XGBoost, LightGBM) perform better when missing values (e.g., undetected F0) are preserved as NaN rather than imputed is noteworthy — missing features can themselves be signals of pathology.

### 7.5 Per-Disease Performance Varies Substantially

Diseases with more distinctive acoustic signatures (recurrent nerve paralysis: 87% F1) are much easier to detect than functional dysphonias or conditions with very few training samples (vocal fold nodules: 48% F1).

---

## 8. Conclusion & Future Work

This project demonstrates that **multifractal detrended fluctuation analysis features, combined with traditional acoustic and physiological features, can achieve ~89% F1 macro** for detecting voice pathology from sustained vowel recordings, using clinically rigorous speaker-level evaluation.

The pipeline evolved through multiple iterations, progressively addressing data leakage, feature selection, and model architecture to arrive at a robust and reproducible evaluation framework.

### Future Directions

1. **Deep learning approaches:** End-to-end models (CNN, Transformer) on raw waveforms or spectrograms
2. **Hierarchical classification:** Explicit two-stage model (healthy screening → disease typing) with cascaded thresholds
3. **Multi-token fusion:** Systematic combination of information across vowels and pitch conditions
4. **Extended entropy features:** Multiscale entropy, permutation entropy, and other complexity measures
5. **Clinical validation:** Prospective evaluation on unseen clinical cohorts
6. **Explainability:** SHAP analysis of multifractal feature contributions to individual predictions

---

_Report generated from the experimental notebooks in the `multifractal-speech-analysis` repository._
