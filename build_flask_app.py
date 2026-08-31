# -*- coding: utf-8 -*-
import os
import shutil

base_dir = "Alzheimer-EEG-CNN-LSTM"

# Ensure all required folders exist
dirs_to_create = [
    os.path.join(base_dir, "templates"),
    os.path.join(base_dir, "static", "css"),
    os.path.join(base_dir, "static", "js"),
    os.path.join(base_dir, "static", "images", "generated"),
    os.path.join(base_dir, "uploads"),
    os.path.join(base_dir, "dataset", "samples"),
    os.path.join(base_dir, "models"),
    os.path.join(base_dir, "results"),
    os.path.join(base_dir, "src")
]
for d in dirs_to_create:
    os.makedirs(d, exist_ok=True)

# ----------------------------------------------------
# 1. config.py
# ----------------------------------------------------
config_code = """# -*- coding: utf-8 -*-
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'alzheimer-eeg-cnn-lstm-secret-key-2026')
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    ALLOWED_EXTENSIONS = {'set', 'fdt', 'csv', 'tsv', 'txt', 'mat'}
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB max limit
    
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    STATIC_DIR = os.path.join(BASE_DIR, 'static')
    GENERATED_PLOTS_DIR = os.path.join(STATIC_DIR, 'images', 'generated')
    SAMPLES_DIR = os.path.join(BASE_DIR, 'dataset', 'samples')
    
    CLASS_NAMES = ['AD', 'HC', 'FTD']
    CLASS_FULL_NAMES = {
        'AD': "Alzheimer's Disease",
        'HC': "Healthy Control",
        'FTD': "Frontotemporal Dementia"
    }
    CLASS_COLORS = {
        'AD': '#e74c3c',
        'HC': '#2ecc71',
        'FTD': '#f39c12'
    }
    
    MODEL_OPTIONS = {
        '80_20': {
            'name': 'CNN-LSTM 80/20 Model (Recommended)',
            'split': '80% Training / 20% Testing',
            'file_pt': 'cnn_lstm_80_20.pt',
            'file_keras': 'cnn_lstm_80_20.keras',
            'accuracy': '100.00%',
            'f1': '100.00%'
        },
        '50_50': {
            'name': 'CNN-LSTM 50/50 Model',
            'split': '50% Training / 50% Testing',
            'file_pt': 'cnn_lstm_50_50.pt',
            'file_keras': 'cnn_lstm_50_50.keras',
            'accuracy': '100.00%',
            'f1': '100.00%'
        },
        '20_80': {
            'name': 'CNN-LSTM 20/80 Model (Low Resource)',
            'split': '20% Training / 80% Testing',
            'file_pt': 'cnn_lstm_20_80.pt',
            'file_keras': 'cnn_lstm_20_80.keras',
            'accuracy': '82.71%',
            'f1': '80.39%'
        }
    }
    DEFAULT_MODEL = '80_20'
"""
with open(os.path.join(base_dir, "config.py"), "w", encoding="utf-8") as f:
    f.write(config_code)

print("config.py created.")

# ----------------------------------------------------
# 2. requirements.txt update
# ----------------------------------------------------
req_code = """Flask>=2.0.0
Werkzeug>=2.0.0
numpy>=1.19.0
pandas>=1.1.0
scipy>=1.5.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
mne>=0.23.0
torch>=1.10.0
"""
with open(os.path.join(base_dir, "requirements.txt"), "w", encoding="utf-8") as f:
    f.write(req_code)

print("requirements.txt updated.")
