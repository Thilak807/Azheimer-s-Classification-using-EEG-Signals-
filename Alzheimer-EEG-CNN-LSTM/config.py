# -*- coding: utf-8 -*-
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
    DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
    SAMPLES_DIR = os.path.join(DATASET_DIR, 'samples')

    
    # Target Disease Stages from Architecture Diagram Box 7:
    # CN: Cognitively Normal, MCI: Mild Cognitive Impairment, AD: Alzheimer's Disease
    CLASS_NAMES = ['AD', 'CN', 'MCI']
    CLASS_ALIASES = {
        'HC': 'CN',
        'CONTROL': 'CN',
        'FTD': 'MCI',
        'DEMENTIA': 'AD'
    }
    CLASS_FULL_NAMES = {
        'AD': "Alzheimer's Disease (AD)",
        'CN': "Cognitively Normal (CN)",
        'MCI': "Mild Cognitive Impairment (MCI)",
        'HC': "Cognitively Normal (CN)",
        'FTD': "Mild Cognitive Impairment (MCI)"
    }
    CLASS_COLORS = {
        'AD': '#e74c3c',   # Red / High Risk
        'CN': '#2ecc71',   # Green / Normal
        'MCI': '#f39c12',  # Amber / Early Warning
        'HC': '#2ecc71',
        'FTD': '#f39c12'
    }
    
    MODEL_OPTIONS = {
        '80_20': {
            'name': 'CNN-LSTM 80/20 Model (Recommended)',
            'split': '80% Training / 20% Testing',
            'file_pt': 'cnn_lstm_80_20.pt',
            'file_keras': 'cnn_lstm_80_20.keras',
            'accuracy': '97.59%',
            'f1': '90.36%',
            'badge': 'Highest Acc'
        },
        'pso_optimized': {
            'name': 'PSO-Optimized CNN-LSTM (Particle Swarm)',
            'split': 'Swarm Tuned (LR, Filters, Units, Dropout)',
            'file_pt': 'cnn_lstm_pso_optimized.pt',
            'file_keras': 'cnn_lstm_pso_optimized.keras',
            'accuracy': '72.43%',
            'f1': '67.85%',
            'badge': 'PSO Tuned'
        },
        '50_50': {
            'name': 'CNN-LSTM 50/50 Model',
            'split': '50% Training / 50% Testing',
            'file_pt': 'cnn_lstm_50_50.pt',
            'file_keras': 'cnn_lstm_50_50.keras',
            'accuracy': '57.34%',
            'f1': '60.51%',
            'badge': 'Balanced'
        },
        '20_80': {
            'name': 'CNN-LSTM 20/80 Model (Low Resource)',
            'split': '20% Training / 80% Testing',
            'file_pt': 'cnn_lstm_20_80.pt',
            'file_keras': 'cnn_lstm_20_80.keras',
            'accuracy': '65.44%',
            'f1': '58.77%',
            'badge': 'Low Data'
        }
    }
    DEFAULT_MODEL = '80_20'

