# PROJECT REPORT

**Alzheimer’s Disease Detection Using EEG Signals via Hybrid CNN–LSTM**

A Deep Learning Framework with Particle Swarm Optimization and a Flask Web Diagnostic Interface

---

**Academic / Technical Project Report**  
**Domain:** Biomedical Signal Processing, Deep Learning, Healthcare Informatics  
**Technology Stack:** Python, PyTorch, MNE, Flask, Bootstrap  
**Dataset:** OpenNeuro `ds004504` (resting-state EEG)  
**Target Classes:** Cognitively Normal (CN) → Mild Cognitive Impairment (MCI) → Alzheimer’s Disease (AD)

---

## Abstract

Alzheimer’s Disease (AD) is a progressive neurodegenerative disorder. Conventional diagnostic tools such as PET imaging and cerebrospinal fluid analysis are costly and invasive. Electroencephalography (EEG) is a non-invasive, relatively inexpensive method that records cortical electrical activity and shows characteristic “EEG slowing” in dementia.

This project implements an end-to-end system that classifies resting-state EEG into three clinical stages, shown in disease-progression order:

1. **Cognitively Normal (CN)**  
2. **Mild Cognitive Impairment (MCI)**  
3. **Alzheimer’s Disease (AD)**

The pipeline includes 19-channel 10–20 montage selection, 0.5–45 Hz bandpass filtering, 50 Hz notch filtering, Independent Component Analysis (ICA) artifact removal, resampling to 128 Hz, 5-second epoching with 50% overlap, and per-channel z-score normalization. A hybrid **1D Convolutional Neural Network (CNN)** extracts local spatial–temporal features; a **Bidirectional LSTM** models longer temporal structure. **Particle Swarm Optimization (PSO)** tunes learning rate, dropout, CNN filter counts, LSTM units, and batch size.

A Flask web application lets a user upload `.set`, `.csv`, `.tsv`, or `.mat` files (or use sample recordings). The Results page reports the predicted stage, a confidence gauge, and three circular probability meters for CN, MCI, and AD. On the recorded PSO-optimized experiment, the model achieved **99.49% accuracy**, **99.57% precision**, **99.49% recall**, and **99.53% F1-score**. The system is intended for **research and education**, not clinical diagnosis.

---

## Table of Contents

1. Introduction  
2. Problem Statement  
3. Objectives  
4. Scope and Limitations  
5. Background and Motivation  
6. Literature Context  
7. Dataset Description  
8. Proposed System Overview  
9. EEG Preprocessing Pipeline  
10. Hybrid CNN–LSTM Architecture  
11. Particle Swarm Optimization  
12. Training Strategy and Evaluation Protocol  
13. Web Application Design  
14. Experimental Results  
15. Discussion  
16. Ethical Considerations and Disclaimer  
17. Conclusion  
18. Future Work  
19. Tools and Technologies  
20. Project File Structure  
21. How to Run the System  
22. References  

---

## 1. Introduction

Dementia, especially Alzheimer’s Disease, is a major global health challenge. Early identification of cognitive decline can support planning, research, and monitoring. EEG captures millisecond-scale neural activity from the scalp. In AD and related conditions, EEG often shows:

- increased **delta** (0.5–4 Hz) and **theta** (4–8 Hz) power  
- reduced posterior **alpha** (8–13 Hz) rhythm  
- disrupted temporal structure across channels  

Manual EEG reading requires expertise and does not scale well. This project builds an automated classifier that:

- reads raw or exported EEG files  
- applies a reproducible preprocessing chain  
- uses a hybrid CNN–LSTM network  
- optionally uses PSO-tuned hyperparameters  
- presents results through a web interface with class probabilities in **CN → MCI → AD** order  

The work combines signal processing, deep learning, optimization, and software engineering into one deployable system.

---

## 2. Problem Statement

Given a resting-state EEG recording (typically 19 channels, International 10–20 system), automatically assign one of three disease stages:

| Stage | Code | Clinical meaning |
| :--- | :--- | :--- |
| Normal | **CN** | Cognitively normal / healthy control pattern |
| Impairment | **MCI** | Mild cognitive impairment (transitional stage) |
| Alzheimer’s | **AD** | Pattern consistent with Alzheimer’s-related EEG slowing |

**Challenges addressed:**

- EEG is noisy (blinks, muscle, 50 Hz power-line interference).  
- Recordings differ in sampling rate, duration, and file format.  
- Classes can be imbalanced.  
- Window-level labels can leak if the same subject appears in both train and test sets.  
- A research model is not useful if clinicians or students cannot run it without writing code.

**Solution:** a subject-aware training pipeline, ICA-based cleaning, hybrid CNN–LSTM classification, PSO hyperparameter search, and a Flask results dashboard with circular probability meters for all three classes.

---

## 3. Objectives

### Primary objectives

1. Design a complete EEG preprocessing pipeline suitable for deep learning.  
2. Implement a hybrid 1D CNN + Bidirectional LSTM classifier for three-class staging (CN, MCI, AD).  
3. Prevent data leakage using **subject-level** train/test partitioning.  
4. Tune key hyperparameters with Particle Swarm Optimization.  
5. Deliver a web application for upload, inference, visualization, and metric reporting.

### Secondary objectives

- Support multiple file formats (EEGLAB `.set`, CSV/TSV, MATLAB `.mat`).  
- Show preprocessing stages (raw → filtered → ICA-cleaned → normalized).  
- Display class probabilities as circular meters in clinical progression order.  
- Provide sample recordings for demonstration without a full dataset download.

---

## 4. Scope and Limitations

**In scope**

- Resting-state scalp EEG classification into CN / MCI / AD.  
- Offline training scripts and online Flask inference.  
- Visualization of waveforms and model metrics.  
- Educational demonstration of CNN–LSTM and PSO.

**Out of scope / limitations**

- Not a certified medical device.  
- Does not replace neurologist interpretation, cognitive testing, MRI, or PET.  
- Does not diagnose mixed dementia, vascular dementia, or psychiatric mimics.  
- High reported accuracy may not generalize to other clinics, devices, or protocols without external validation.  
- Perfect or near-perfect scores on a small or homogeneous set should be interpreted with caution (possible residual leakage, easy synthetic data, or limited subject diversity).

---

## 5. Background and Motivation

### 5.1 Clinical motivation

AD progresses over years. **MCI** is an intermediate stage with measurable cognitive change but preserved daily function relative to dementia. Detecting a shift from CN toward MCI, and from MCI toward AD, is more useful than a binary “disease vs. healthy” label. The user interface therefore always lists:

**Normal (CN) → Impairment (MCI) → Alzheimer’s (AD)**

### 5.2 Why EEG?

| Modality | Invasiveness | Cost | Temporal resolution | Typical use |
| :--- | :--- | :--- | :--- | :--- |
| PET / CSF | High | High | Low | Gold-standard biomarkers |
| MRI | Low | High | Low | Structure / atrophy |
| EEG | Very low | Lower | Very high | Functional cortical rhythms |

EEG is portable and repeatable. Deep learning can learn filters that correspond to morphological transients and longer oscillatory patterns without hand-crafted features only.

### 5.3 Why hybrid CNN–LSTM?

- **CNN (1D):** local kernels detect spikes, bursts, and short spectral patterns across time; stacking channels treats the montage as a multivariate series.  
- **LSTM:** after pooling, the sequence of CNN feature maps still has temporal order; bidirectional LSTM captures context in both directions.  
- **Softmax head:** outputs a probability for each of the three stages.

---

## 6. Literature Context

Typical published pipelines for EEG–dementia classification include:

1. **Hand-crafted features** — band power, coherence, entropy, Hjorth parameters, then SVM/Random Forest.  
2. **CNN-only models** — good at local patterns, weaker at long-range sequence memory.  
3. **RNN/LSTM-only models** — sequence modeling without strong local feature extractors.  
4. **Hybrid CNN–RNN** — the approach used here.  
5. **Hyperparameter search** — grid search is expensive; PSO and related swarm methods sample a mixed continuous/discrete space (learning rate, filters, units, dropout).

This project follows the hybrid CNN–LSTM line and adds ICA cleaning, subject-level splits, PSO, and a production-style web UI.

---

## 7. Dataset Description

### 7.1 Primary public dataset

Training and evaluation are designed around **OpenNeuro dataset ds004504** (resting-state EEG related to Alzheimer’s and control groups). The loader maps subjects into three groups used by this application:

| Group | Subject ID range (project mapping) | Meaning |
| :--- | :--- | :--- |
| AD | 1–36 | Alzheimer’s Disease |
| CN | 37–65 | Cognitively Normal |
| MCI | 66–88 | Mild Cognitive Impairment |

Aliases are supported so older labels still map correctly:

- `HC` / `CONTROL` → **CN**  
- `FTD` → **MCI** (application-level staging alias)  

### 7.2 Recording characteristics

- **Montage:** 19 channels, International 10–20  
  `Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, Fz, Cz, Pz`  
- **Typical original rate:** often 500 Hz (resampled to **128 Hz**)  
- **Input tensor after windowing:** `(640 samples × 19 channels)` for a 5.0 s window at 128 Hz  

### 7.3 Related historical EEG collection (MATLAB-era notes)

The repository also documents an older tabular EEG collection (ACE-R style scores) with AD, MCI, and control labels and mixed 256 Hz / 128 Hz sampling. That collection is **not** the same as OpenNeuro `ds004504`. The deep-learning application is built around the OpenNeuro-style 19-channel pipeline.

### 7.4 Demo samples

The web app can generate or ship CSV samples such as:

- Cognitively Normal  
- Mild Cognitive Impairment  
- Alzheimer’s  

so the UI can be demonstrated without downloading the full BIDS tree.

---

## 8. Proposed System Overview

End-to-end data flow:

```
EEG file (.set / .csv / .tsv / .mat)
        │
        ▼
Parse & load (MNE / pandas / scipy)
        │
        ▼
Channel pick (19 × 10–20)
        │
        ▼
Bandpass 0.5–45 Hz + Notch 50 Hz
        │
        ▼
ICA artifact removal (blink / muscle)
        │
        ▼
Resample 128 Hz + clip + z-score
        │
        ▼
5 s epochs, 50% overlap  →  (N, 640, 19)
        │
        ▼
Hybrid CNN–LSTM  (optional PSO weights)
        │
        ▼
Mean softmax over windows
        │
        ▼
Web result: CN / MCI / AD circular meters
```

**Design principles**

- Same preprocessing at train time and at inference time.  
- Window-level predictions are **averaged** so a long recording becomes one subject-level decision.  
- UI order follows clinical progression, independent of the model’s internal class index order (`AD=0, CN=1, MCI=2`).

---

## 9. EEG Preprocessing Pipeline

Implemented in `src/preprocessing.py`.

### Stage 1 — Channel selection

Only standard 10–20 names are kept. Missing channels are padded to 19 at inference so the network always sees a fixed width.

### Stage 2 — Bandpass filter (0.5–45 Hz)

A Butterworth-style IIR bandpass (via MNE) retains delta through low-gamma content used in dementia EEG studies and removes slow drift and high-frequency noise.

### Stage 3 — Notch filter (50 Hz)

Removes power-line hum (50 Hz; skipped if sampling rate is too low for a valid notch).

### Stage 4 — ICA artifact removal

FastICA (MNE, with a scikit-learn fallback) estimates independent components. Components correlated with frontal channels (`Fp1`, `Fp2`, `F7`, `F8`) are treated as likely ocular artifact. High-kurtosis components are treated as spike/muscle-like. A limited number of components (at most about one quarter of ICA components) are zeroed so brain signal is not over-stripped.

### Stage 5 — Resample, clip, normalize

- Resample to **128 Hz** (5 s → 640 samples).  
- Clip values beyond ±5 standard deviations.  
- **Z-score each channel** independently.

### Epoching (`src/segmentation.py`)

- Window length: **5.0 seconds**  
- Overlap: **50%**  
- Output shape per window: `(window_samples, n_channels)` for CNN/LSTM  

This yields many samples per subject while keeping temporal context inside each window.

---

## 10. Hybrid CNN–LSTM Architecture

Implemented in `src/model.py` (PyTorch primary; Keras variant available).

### 10.1 Block diagram (conceptual)

```
Input: [batch, 640 time, 19 channels]
        │
        ▼  transpose to [batch, 19, 640]
Conv1D  filters_1, k=7  → BN → ReLU → MaxPool2 → Dropout
Conv1D  filters_2, k=5  → BN → ReLU → MaxPool2 → Dropout
Conv1D  filters_2, k=3  → BN → ReLU → MaxPool2 → Dropout
        │
        ▼  [batch, reduced_time, filters_2]
Bidirectional LSTM (lstm_units)
        │  last time step
Dense 64 → BN → ReLU → Dropout
        │
        ▼
Linear → 3 logits → Softmax
        │
        ▼
P(AD), P(CN), P(MCI)
```

Default (non-PSO) sizes: `filters_1=64`, `filters_2=128`, `lstm_units=64`, `dropout=0.3`.

### 10.2 Role of each part

| Module | Role |
| :--- | :--- |
| Conv k=7 | Wider temporal receptive field (slower waves, envelopes) |
| Conv k=5 / k=3 | Finer morphology |
| Max pooling | Time compression; translation tolerance |
| BatchNorm + Dropout | Regularization |
| Bi-LSTM | Sequence memory after CNN compression |
| Softmax | Calibrated three-way probabilities (in-sample) |

### 10.3 Loss and optimizer

- Loss: **cross-entropy** (class-weighted during training to reduce imbalance).  
- Optimizer: **Adam** (weight decay `1e-4` in PyTorch).  
- Early stopping on validation loss (patience 12).

---

## 11. Particle Swarm Optimization

Implemented in `src/pso_optimizer.py`. PSO treats each “particle” as a candidate hyperparameter vector. Particles move using personal best (`pBest`) and global best (`gBest`).

### 11.1 Search space

| Dimension | Mapping |
| :--- | :--- |
| Learning rate | log-scale \(10^{-4}\) to \(10^{-2}\) |
| Dropout | 0.15–0.50 |
| CNN filters block 1 | {32, 64, 96, 128} |
| CNN filters block 2–3 | {64, 128, 192, 256} |
| LSTM units | {32, 64, 96, 128} |
| Batch size | {16, 32, 64} |

Typical swarm settings in code: 6 particles, few iterations, short evaluation epochs (fitness is validation-oriented). Inertia decays from 0.9 toward 0.4; cognitive/social coefficients \(c_1 = c_2 = 1.494\).

### 11.2 Best hyperparameters found (saved)

From `results/pso_optimal_hyperparameters.json`:

| Hyperparameter | Optimal value |
| :--- | :--- |
| Learning rate | 0.003698 |
| Dropout | 0.227 |
| CNN filters 1 | 96 |
| CNN filters 2 | 192 |
| LSTM units | 64 |
| Batch size | 64 |
| Best swarm fitness | 0.8374 (during search) |

Fitness during search is not identical to the final fully trained test metrics. After full training with these settings, test metrics are reported in Section 14.

---

## 12. Training Strategy and Evaluation Protocol

### 12.1 Subject-level split

Windows from the **same subject must not** appear in both training and testing. `segmentation.py` builds windows after assigning whole subjects to train or test. This is essential for EEG: overlapping windows from one person are highly correlated.

### 12.2 Class imbalance

`sklearn.utils.class_weight.compute_class_weight(mode='balanced')` scales the loss so minority stages (often MCI) are not ignored.

### 12.3 Metrics

Reported as **macro-averaged** (equal weight per class):

- Accuracy  
- Precision  
- Recall (sensitivity)  
- F1-score  

Confusion matrices and training curves are stored under `results/` and shown on the Dashboard page.

### 12.4 Inference aggregation

For an uploaded file:

1. Extract all valid 5 s windows.  
2. Softmax each window.  
3. **Mean** the probability vectors.  
4. Predicted class = argmax of the mean.  
5. Confidence = that class’s mean probability × 100%.  

UI then shows CN, MCI, and AD percentages on three circular meters (not only the winning class).

---

## 13. Web Application Design

Implemented with **Flask** (`app.py`), Bootstrap 5 templates, and `static/css/style.css`.

### 13.1 Routes

| Route | Purpose |
| :--- | :--- |
| `/` | Home: problem, architecture highlights, class cards (CN → MCI → AD) |
| `/upload` | Drag-and-drop upload, model choice, live scan animation, AJAX predict |
| `/predict` | Server-side preprocess + CNN–LSTM inference (JSON or redirect) |
| `/result` | Diagnosis banner, confidence ring, three class meters, EEG stage plot |
| `/dashboard` | Accuracy / precision / recall / F1, comparison table, PSO panel |
| `/about` | Clinical motivation and architecture explanation |

### 13.2 Results page (client requirements)

Two UI rules are applied throughout:

1. **Order:** Normal (CN), then Impairment (MCI), then Alzheimer’s (AD).  
2. **Meters:** each of the three class probabilities uses the **same circular gauge** (green / amber / red), not a mix of one ring and two bars.

The large top ring remains **prediction confidence**. The three smaller rings are **class probabilities**.

### 13.3 Security and operations notes

- Uploads are size-capped (100 MB) and extension-filtered.  
- Filenames are sanitized (`secure_filename`).  
- The app is a local research demo; it should not be exposed to the public internet without authentication, HTTPS, and a clinical governance process.

---

## 14. Experimental Results

### 14.1 PSO-optimized model (primary recorded result)

Source: `results/model_results.csv`

| Configuration | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) |
| :--- | ---: | ---: | ---: | ---: |
| PSO-optimized CNN–LSTM | **99.49** | **99.57** | **99.49** | **99.53** |

These figures are window- or sample-level scores from the project’s saved evaluation table. They indicate very strong in-project discrimination among CN, MCI, and AD under the training protocol used.

### 14.2 Additional split experiments (documented in README / config)

The application also ships configuration entries for other train/test ratios (subject-level):

| Split | Role | Config-listed accuracy | Config-listed F1 |
| :--- | :--- | :--- | :--- |
| 80 / 20 | Standard high-resource split | 100.00% | 100.00% |
| 50 / 50 | Balanced partition | 100.00% | 100.00% |
| 20 / 80 | Low-data / harder generalization | 82.71% | 80.39% |
| PSO-optimized | Swarm-tuned architecture | 100.00% (config label) / **99.49% (CSV)** | 100.00% / **99.53%** |

**Interpretation:** 100% on a small test set can mean the test subjects were easy to separate, the set was small, or evaluation was not fully external. The **20/80** drop to ~83% accuracy is a more realistic reminder that performance depends on how much unique subject data the model has seen. The CSV value **99.49%** is the most concrete saved metric for the PSO run.

### 14.3 PSO search trace

Saved gBest fitness by iteration (short search):

| Iteration | gBest fitness | Average swarm fitness |
| ---: | ---: | ---: |
| 1 | 0.8011 | 0.6088 |
| 2 | 0.8011 | 0.5857 |
| 3 | 0.8374 | 0.6926 |

Plots: `results/pso_convergence.png`, `results/training_history_pso_optimized.png`, `results/confusion_matrix_pso_optimized.png`.

---

## 15. Discussion

**What worked**

- Combining CNN local filters with LSTM sequence modeling matches EEG’s dual nature (waveform shape + evolving rhythm).  
- ICA plus notch/bandpass is a standard clinical-research preprocessing stack.  
- Subject-level splitting is the correct scientific choice even if it lowers headline accuracy.  
- A web UI with three equal circular meters makes the **full probability distribution** visible, which is more honest than showing only the winning class.  
- PSO found a higher-capacity CNN (96/192 filters) and a slightly lower dropout (0.227) with a larger batch size (64).

**What to treat carefully**

- Near-perfect metrics are uncommon on truly unseen, multi-site EEG. Independent replication is required before any clinical claim.  
- Mapping historical FTD labels to MCI is an **application convention**, not a statement that FTD equals MCI.  
- Mean-pooling windows can hide a minority of contradictory epochs.  
- Demo/synthetic samples can look “too clean” compared with real clinic EEG.

**Engineering trade-offs**

- 128 Hz and 5 s windows balance compute and context.  
- ICA can fail on very short or few-channel files; a fallback path exists.  
- Dual Keras/PyTorch wrappers increase maintenance cost but ease environment differences.

---

## 16. Ethical Considerations and Disclaimer

- EEG contains health-related information. Store uploads only as needed; do not publish identifiable recordings.  
- Model error can be **false reassurance** (missed AD/MCI) or **false alarm** (healthy labeled as AD).  
- Bias may arise if the training cohort is not diverse in age, sex, ethnicity, medications, or EEG hardware.  

**Disclaimer:** This software is for **academic research and education**. It is **not** a medical diagnosis, **not** FDA/CE cleared, and **must not** be used as the sole basis for clinical decisions. Diagnosis of Alzheimer’s disease and MCI requires qualified clinicians and established diagnostic criteria.

---

## 17. Conclusion

This project delivers a complete Alzheimer’s-related EEG classification system:

- a clinically motivated **three-stage** output (CN → MCI → AD)  
- a documented **five-stage preprocessing** chain including ICA  
- a **hybrid CNN–LSTM** model with optional **PSO** hyperparameter search  
- **subject-level** experimental protocol and saved metrics (**99.49% accuracy** on the PSO evaluation table)  
- a **Flask** interface whose Results page shows **three circular probability meters** in progression order  

The main scientific contribution is the integration of signal processing, deep sequence modeling, swarm-based tuning, and a usable diagnostic-style UI—not a claim of clinical replacement for standard dementia work-up.

---

## 18. Future Work

1. **External validation** on independent EEG datasets and different amplifiers.  
2. **Leave-one-subject-out** and nested cross-validation with published confidence intervals.  
3. **Explainability:** saliency or class-activation maps over time and channels (e.g., posterior alpha).  
4. **Longitudinal modeling:** change from CN toward MCI over repeated sessions.  
5. **Calibration:** temperature scaling so on-screen percentages match true frequencies.  
6. **Richer clinical fusion:** age, MMSE/MoCA, MRI features.  
7. **Deployment:** GPU serving, authentication, audit logs, DICOM/BIDS ingest.  
8. **Fairness audit:** error rates stratified by age and sex.

---

## 19. Tools and Technologies

| Layer | Tools |
| :--- | :--- |
| Language | Python 3 |
| Deep learning | PyTorch (primary), TensorFlow/Keras (optional) |
| EEG I/O & filters | MNE-Python, SciPy |
| ML metrics / splits | scikit-learn |
| Optimization | Custom PSO |
| Web | Flask, Jinja2, Bootstrap 5, JavaScript |
| Numerics / plots | NumPy, Pandas, Matplotlib, Seaborn |
| Formats | EEGLAB SET, CSV/TSV, MAT |

Key packages (`requirements.txt`): Flask, NumPy, Pandas, SciPy, Matplotlib, Seaborn, scikit-learn, MNE, PyTorch.

---

## 20. Project File Structure (simplified)

```
Alzheimer-EEG-CNN-LSTM/
├── app.py                      # Flask application
├── config.py                   # Classes, colors, model files, limits
├── requirements.txt
├── run_pipeline.py             # Training / benchmark
├── run_pso_optimization.py
├── train_full_models.py
├── PROJECT_REPORT.md           # This report
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── model.py
│   ├── train.py
│   ├── evaluation.py
│   ├── pso_optimizer.py
│   └── prediction.py
├── templates/                  # index, upload, result, dashboard, about
├── static/css/style.css
├── models/                     # .pt / .keras checkpoints
├── results/                    # CSV, confusion matrices, PSO JSON
└── dataset/                    # EEG data + samples
```

---

## 21. How to Run the System

```powershell
cd Alzheimer-EEG-CNN-LSTM
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`, go to **Upload EEG**, choose a sample or a recording, and inspect **Results** (CN → MCI → AD circular meters).

Offline training (when the full dataset is present):

```powershell
python run_pipeline.py
python run_pso_optimization.py
```

---

## 22. References

1. OpenNeuro Dataset **ds004504** — resting-state EEG related to Alzheimer’s disease and control subjects. https://openneuro.org/datasets/ds004504  
2. International 10–20 system of electrode placement (Jasper, 1958; later ILAE/IFCN updates).  
3. LeCun, Y., Bengio, Y., & Hinton, G. Deep learning. *Nature*, 2015.  
4. Hochreiter, S., & Schmidhuber, J. Long Short-Term Memory. *Neural Computation*, 1997.  
5. Graves, A., & Schmidhuber, J. Framewise phoneme classification with bidirectional LSTM. *Neural Networks*, 2005.  
6. Kennedy, J., & Eberhart, R. Particle Swarm Optimization. *IEEE ICNN*, 1995.  
7. Hyvärinen, A., & Oja, E. Independent Component Analysis: algorithms and applications. *Neural Networks*, 2000.  
8. MNE-Python documentation — filtering, ICA, resampling. https://mne.tools  
9. Cassani, R., et al. Systematic review on EEG-based biomarkers for Alzheimer’s disease. (representative EEG–AD review literature).  
10. PyTorch documentation. https://pytorch.org  
11. Flask documentation. https://flask.palletsprojects.com  

*(Students should add IEEE/APA citations from papers actually read, plus any college-specific format for aim, literature survey, and Gantt charts.)*

---

## Appendix A — Class Index Mapping (model vs UI)

| Softmax index | Internal code | UI label | UI order |
| ---: | :--- | :--- | ---: |
| 0 | AD | Alzheimer’s (AD) | 3 |
| 1 | CN | Normal (CN) | 1 |
| 2 | MCI | Impairment (MCI) | 2 |

The network is **not** retrained to change index order. Only display order and meter design change.

## Appendix B — Default vs PSO architecture sizes

| Parameter | Default CNN–LSTM | PSO-optimized |
| :--- | ---: | ---: |
| CNN filters 1 | 64 | 96 |
| CNN filters 2 | 128 | 192 |
| LSTM units | 64 | 64 |
| Dropout | 0.30 | 0.227 |
| Learning rate | 0.001 | 0.003698 |
| Batch size | 32 | 64 |

## Appendix C — Suggested viva / defense questions

1. Why subject-level split instead of random windows?  
2. Why 5 seconds and 128 Hz?  
3. What does ICA remove, and what can it accidentally remove?  
4. Why bidirectional LSTM after CNN rather than CNN only?  
5. How is one recording turned into a single diagnosis?  
6. Why show all three probabilities, not only the predicted class?  
7. Why might 99% accuracy still be scientifically weak?  
8. Is this system clinically approved? (Answer: no.)

---

**End of Report**
