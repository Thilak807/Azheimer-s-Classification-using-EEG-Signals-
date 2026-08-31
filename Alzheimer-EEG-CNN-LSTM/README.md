# Alzheimer's Disease Detection Using EEG Signals via Hybrid CNN-LSTM
## Complete Flask Web Application & Deep Learning Diagnostic System

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Web Framework](https://img.shields.io/badge/backend-Flask%202.x-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Deep Learning](https://img.shields.io/badge/deep--learning-CNN--LSTM-orange.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/dataset-OpenNeuro%20ds004504-green.svg)](https://openneuro.org/datasets/ds004504)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 1. Project Title & Overview
**Project Title**: *Alzheimer's Disease Detection Using EEG Signals Using Hybrid CNN-LSTM*

An automated, non-invasive deep learning diagnostic framework integrated into a modern **Flask Web Application** for multi-class classification of resting-state Electroencephalography (EEG) recordings into:
1. **AD (Alzheimer's Disease)**
2. **HC (Healthy Control)**
3. **FTD (Frontotemporal Dementia)**

---

## 2. Problem Statement & Clinical Motivation
Alzheimer's Disease (AD) causes progressive cognitive deterioration and memory impairment. Standard diagnostic modalities (PET scans, spinal taps) are expensive and invasive. Scalp EEG offers an accessible, non-invasive window into neuro-electrophysiology:
- **EEG Slowing**: Widespread power increase in low-frequency **Delta (0.5–4 Hz)** and **Theta (4–8 Hz)** bands.
- **Alpha Attenuation**: Significant reduction in dominant posterior **Alpha (8–13 Hz)** waves.
- **Frontal Frequency Shifts**: Frontotemporal Dementia (FTD) displays localized anterior abnormalities.

---

## 3. Web Application Features & Architecture

### User Flow
```
Browser User
    │
    ▼
[Web Portal]: http://127.0.0.1:5000 (Home / Upload / Dashboard / About)
    │
    ▼
[Upload EEG]: Upload .set, .csv, .tsv, .mat or Pick Preloaded Sample
    │
    ▼
[Signal Processing]: 0.5–45 Hz Bandpass + 50 Hz Notch + 128 Hz Resample + Z-Score
    │
    ▼
[Segmentation]: 5.0-Second Sliding Windows (640 Samples x 19 Channels)
    │
    ▼
[Model Inference]: Saved Pre-Trained Hybrid CNN-LSTM Model (Zero Training at Runtime)
    │
    ▼
[Result Visualization]: Predicted Class + Confidence Meter + Probabilities + Waveform Plot
```

---

## 4. Flask Routes & Pages

| Route | Page | Description |
| :--- | :--- | :--- |
| `/` | **Home Page** | Hero overview, 3 feature cards, quick actions, pipeline summary. |
| `/upload` | **EEG Upload** | Drag-and-drop file upload, model selector (50/50, 80/20, 20/80), 1-click clinical samples. |
| `/predict` | **Inference Engine** | Multi-step preprocessing, windowing, and CNN-LSTM prediction. |
| `/result` | **Diagnosis Result** | Predicted diagnosis badge, confidence meter, probability breakdown, Raw vs Preprocessed EEG wave plot, clinical disclaimer. |
| `/dashboard` | **Performance Dashboard** | Real-time metric cards (Accuracy, Precision, Recall, F1), comparison table from `model_results.csv`, tabbed confusion matrices and training curves. |
| `/about` | **Technical Guide** | Electrophysiology background, CNN and LSTM roles, evaluation metrics guide. |

---

## 5. Experimental Results (Subject-Level Partitioning)

Results from `results/model_results.csv` evaluated on untouched test subjects:

| Train/Test Split | Training Subjects | Testing Subjects | Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro F1-Score (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **80 / 20** | 70 subjects | 18 subjects | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| **50 / 50** | 44 subjects | 44 subjects | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| **20 / 80** | 17 subjects | 71 subjects | **82.71%** | **86.55%** | **82.21%** | **80.39%** |

---

## 6. Directory Structure

```
Alzheimer-EEG-CNN-LSTM/
│
├── app.py                                   # Main Flask Web Application
├── config.py                                # Configuration settings
├── requirements.txt                         # Package dependencies
├── run_pipeline.py                          # Offline model training & benchmark script
├── README.md                                # Project documentation
│
├── dataset/                                 # Dataset root (OpenNeuro ds004504)
│   ├── participants.tsv                     # Subject clinical metadata
│   ├── derivatives/                         # 19-channel EEG recordings
│   └── samples/                             # 1-Click sample EEG files for web demo
│       ├── sample_AD_Alzheimer_subject.csv
│       ├── sample_HC_Healthy_subject.csv
│       └── sample_FTD_Dementia_subject.csv
│
├── models/                                  # Saved model checkpoints
│   ├── cnn_lstm_50_50.pt (or .keras)
│   ├── cnn_lstm_80_20.pt (or .keras)
│   └── cnn_lstm_20_80.pt (or .keras)
│
├── src/                                     # Modular Python package
│   ├── __init__.py
│   ├── data_loader.py                       # Dataset loader & OpenNeuro inspection
│   ├── preprocessing.py                     # 0.5-45Hz Bandpass, 50Hz Notch, 128Hz Resample
│   ├── segmentation.py                      # 5s windowing (~1000 samples) & subject-level split
│   ├── model.py                             # Hybrid CNN-LSTM architecture
│   ├── train.py                             # Training engine with class weights
│   ├── evaluation.py                        # Metrics, confusion heatmaps & curve plotting
│   └── prediction.py                        # Inference engine & signal visualization
│
├── results/                                 # Metric tables & plots
│   ├── model_results.csv                    # Final comparison table
│   ├── confusion_matrix_50_50.png
│   ├── confusion_matrix_80_20.png
│   ├── confusion_matrix_20_80.png
│   ├── training_history_50_50.png
│   ├── training_history_80_20.png
│   └── training_history_20_80.png
│
├── templates/                               # HTML Templates
│   ├── base.html                            # Navbar, footer, Bootstrap 5 theme
│   ├── index.html                           # Home landing page & feature cards
│   ├── upload.html                          # Drag & drop upload + sample selector
│   ├── result.html                          # Diagnosis, confidence & EEG wave plot
│   ├── dashboard.html                       # Model comparison table & charts
│   ├── about.html                           # Scientific & technical documentation
│   └── error.html                           # Graceful error handler
│
├── static/                                  # Static assets
│   ├── css/style.css                        # Modern healthcare AI theme
│   ├── js/app.js                            # Upload interactions & progress
│   └── images/results/                      # Confusion matrix & curve plots
│
└── uploads/                                 # Temporary upload storage
```

---

## 7. How to Install & Run

### Step 1: Navigate to Project Directory
```powershell
cd "c:\Users\Lenovo\Downloads\Azheimer-s-Classification-using-EEG-Signals-master\Alzheimer-EEG-CNN-LSTM"
```

### Step 2: Create & Activate Virtual Environment
```powershell
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Run the Flask Web Application
```powershell
python app.py
```

### Step 5: Open in Browser
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 8. Clinical Research Disclaimer
*This system provides a machine-learning prediction for research and educational purposes and is not a clinical medical diagnosis. Clinical diagnosis must be performed by certified neurologists and healthcare professionals.*
