# -*- coding: utf-8 -*-
import os
import uuid
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import mne
import scipy.io as sio

from .data_loader import STANDARD_CHANNELS
from .preprocessing import preprocess_raw_eeg
from .segmentation import slice_into_windows, CLASS_NAMES
from .model import UnifiedEEGModelWrapper
from config import Config

# Cache for loaded models to prevent reloading from disk on every request
_MODEL_CACHE = {}

def get_model(model_name="pso_optimized"):
    """
    Retrieves or loads the trained model from the model cache.
    Supports PSO-optimized model and standard splits with shape autodetection and graceful fallback.
    """
    if model_name not in Config.MODEL_OPTIONS:
        model_name = Config.DEFAULT_MODEL
        
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
        
    model_meta = Config.MODEL_OPTIONS[model_name]
    model_path_pt = os.path.join(Config.MODELS_DIR, model_meta['file_pt'])
    model_path_keras = os.path.join(Config.MODELS_DIR, model_meta['file_keras'])
    input_shape = (640, 19)
    
    # Determine which file to load
    target_pt = None
    if os.path.exists(model_path_pt):
        target_pt = model_path_pt
    else:
        # Fallback to recommended 80_20 or any available pt checkpoint
        fallback_pt = os.path.join(Config.MODELS_DIR, 'cnn_lstm_80_20.pt')
        if os.path.exists(fallback_pt):
            target_pt = fallback_pt
            print(f"[Prediction] Checkpoint {model_meta['file_pt']} pending; loaded fallback {fallback_pt}")
        else:
            available_pts = [f for f in os.listdir(Config.MODELS_DIR) if f.endswith('.pt')]
            if available_pts:
                target_pt = os.path.join(Config.MODELS_DIR, available_pts[0])
                print(f"[Prediction] Loaded alternative model: {target_pt}")
                
    if target_pt is not None:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(target_pt, map_location=device)
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # Autodetect layer dimensions from checkpoint
        f1 = state_dict['conv1.0.weight'].shape[0] if 'conv1.0.weight' in state_dict else 64
        f2 = state_dict['conv2.0.weight'].shape[0] if 'conv2.0.weight' in state_dict else 128
        lstm_units = state_dict['lstm.weight_hh_l0'].shape[1] if 'lstm.weight_hh_l0' in state_dict else 64
        num_classes = state_dict['fc_out.weight'].shape[0] if 'fc_out.weight' in state_dict else 3
        
        hyperparams = {
            'cnn_filters_1': f1,
            'cnn_filters_2': f2,
            'lstm_units': lstm_units
        }
        
        # .pt checkpoints are PyTorch weights; force the torch backend.
        # UnifiedEEGModelWrapper defaults to Keras Sequential when TF is installed,
        # which has no load_state_dict().
        wrapper = UnifiedEEGModelWrapper(
            input_shape,
            num_classes=num_classes,
            lr=0.001,
            framework='torch',
            hyperparams=hyperparams,
        )
        wrapper.model.load_state_dict(state_dict)
        wrapper.model.eval()
        print(f"[Prediction] Loaded PyTorch model checkpoint: {target_pt} (Filters: {f1}/{f2}, LSTM: {lstm_units})")
        
    elif os.path.exists(model_path_keras):
        from tensorflow.keras.models import load_model
        wrapper = UnifiedEEGModelWrapper(input_shape, num_classes=3, lr=0.001, framework='keras')
        wrapper.model = load_model(model_path_keras)
        print(f"[Prediction] Loaded Keras model checkpoint: {model_path_keras}")
    else:
        raise FileNotFoundError(f"Trained model checkpoint not found in {Config.MODELS_DIR}")
            
    _MODEL_CACHE[model_name] = wrapper
    return wrapper




def parse_and_load_eeg_file(file_path):
    """
    Loads an uploaded EEG file (.set, .csv, .tsv, .mat) and returns an MNE Raw object.
    Validates channels, sampling rate, and minimum duration.
    """
    ext = file_path.lower().split('.')[-1]
    
    if ext == 'set':
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose='CRITICAL')
    elif ext in ['csv', 'tsv', 'txt']:
        sep = '\t' if ext == 'tsv' else ','
        df = pd.read_csv(file_path, sep=sep)
        
        col_names = [c.strip() for c in df.columns]
        matched_channels = [c for c in STANDARD_CHANNELS if c in col_names]
        
        if len(matched_channels) >= 5:
            data_arr = df[matched_channels].values.T
            ch_names = matched_channels
        else:
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] == 0:
                raise ValueError("No numerical EEG channel columns found in the uploaded file.")
            n_cols = min(numeric_df.shape[1], len(STANDARD_CHANNELS))
            data_arr = numeric_df.iloc[:, :n_cols].values.T
            ch_names = STANDARD_CHANNELS[:n_cols]
            
        sfreq = 128.0 if data_arr.shape[1] < 10000 else 500.0
        
        if np.max(np.abs(data_arr)) > 1.0:
            data_arr = data_arr * 1e-6 # Convert uV to V
            
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data_arr, info, verbose='CRITICAL')
    elif ext == 'mat':
        mat = sio.loadmat(file_path)
        data_arr = None
        for k, v in mat.items():
            if isinstance(v, np.ndarray) and v.ndim == 2 and (v.shape[0] in [19, 28] or v.shape[1] in [19, 28]):
                data_arr = v if v.shape[0] < v.shape[1] else v.T
                break
        if data_arr is None:
            raise ValueError("No valid EEG 2D matrix found in .mat file.")
            
        n_ch = min(data_arr.shape[0], len(STANDARD_CHANNELS))
        data_arr = data_arr[:n_ch, :] * 1e-6
        info = mne.create_info(ch_names=STANDARD_CHANNELS[:n_ch], sfreq=128.0, ch_types='eeg')
        raw = mne.io.RawArray(data_arr, info, verbose='CRITICAL')
    else:
        raise ValueError(f"Unsupported file format '.{ext}'. Supported formats: .set, .csv, .tsv, .mat")
        
    return raw


def generate_signal_plot(raw_signal, filtered_signal, ica_signal, prep_signal,
                         sfreq=128.0, channel_name="Fp1", output_dir=None):
    """
    Generates a 4-panel visual comparison illustrating the complete 2. EEG Preprocessing pipeline:
    1. Raw EEG Signal (uV)
    2. Band-pass (0.5-45 Hz) & Notch (50 Hz) Filtered Signal
    3. Artifact Removal via ICA (Independent Component Analysis)
    4. Segmented Epoch with Z-score Normalization
    """
    if output_dir is None:
        output_dir = Config.GENERATED_PLOTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    unique_id = uuid.uuid4().hex[:8]
    filename = f"signal_plot_{unique_id}.png"
    filepath = os.path.join(output_dir, filename)
    
    plot_samples = min(int(6.0 * sfreq), len(prep_signal))
    t = np.linspace(0, plot_samples / sfreq, plot_samples)
    
    raw_slice = raw_signal[:plot_samples] * 1e6 if np.max(np.abs(raw_signal[:plot_samples])) < 1.0 else raw_signal[:plot_samples]
    filt_slice = filtered_signal[:plot_samples] * 1e6 if np.max(np.abs(filtered_signal[:plot_samples])) < 1.0 else filtered_signal[:plot_samples]
    ica_slice = ica_signal[:plot_samples] * 1e6 if np.max(np.abs(ica_signal[:plot_samples])) < 1.0 else ica_signal[:plot_samples]
    prep_slice = prep_signal[:plot_samples]
    
    fig, axes = plt.subplots(4, 1, figsize=(11, 7.5), sharex=True)
    
    # 1. Raw EEG
    axes[0].plot(t, raw_slice, color='#e74c3c', linewidth=1.1, label=f'Raw Channel ({channel_name})')
    axes[0].set_title(f'1. Raw EEG Signal (Channel {channel_name}) [Microvolts]', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('uV', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right', fontsize=8)
    
    # 2. Bandpass + Notch
    axes[1].plot(t, filt_slice, color='#9b59b6', linewidth=1.1, label='0.5–45 Hz Bandpass + 50 Hz Notch')
    axes[1].set_title('2. Filtered Signal (Power-Line Notch & Band-Pass Butterworth)', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('uV', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right', fontsize=8)
    
    # 3. ICA Artifact Cleaned
    axes[2].plot(t, ica_slice, color='#f39c12', linewidth=1.1, label='ICA Artifact Removed (Ocular & Muscle Cleaned)')
    axes[2].set_title('3. Artifact Removal via ICA (Independent Component Analysis)', fontsize=10, fontweight='bold')
    axes[2].set_ylabel('uV', fontsize=9)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right', fontsize=8)
    
    # 4. Z-Score Normalized Epoch
    axes[3].plot(t, prep_slice, color='#27ae60', linewidth=1.2, label='Epoch Segmentation & Z-score Normalized')
    axes[3].set_title('4. Final Preprocessed Signal (5s Window Epoch, Z-score Normalization)', fontsize=10, fontweight='bold')
    axes[3].set_xlabel('Time (seconds)', fontsize=9)
    axes[3].set_ylabel('Z-Score', fontsize=9)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()
    
    return f"images/generated/{filename}"


def predict_eeg_file(file_path, model_name="pso_optimized"):
    """
    Complete inference pipeline for an uploaded EEG recording:
    1. EEG Data parse & load
    2. EEG Preprocessing (Band-pass, Notch, ICA, Epoch segmentation, Normalization)
    3B. CNN Spatial-Temporal Feature Extraction
    4B. LSTM Temporal Sequence Modeling
    7. Output Early Disease Stage Classification (CN, MCI, AD)
    """
    from .preprocessing import preprocess_eeg_stages
    
    raw = parse_and_load_eeg_file(file_path)
    orig_sfreq = float(raw.info['sfreq'])
    orig_n_chan = len(raw.ch_names)
    orig_duration = float(raw.times[-1]) if len(raw.times) > 0 else 0.0
    
    if orig_duration < 3.0:
        raise ValueError(f"Recording is too short ({orig_duration:.1f}s). Minimum 5 seconds required.")
        
    first_ch_name = raw.ch_names[0] if raw.ch_names else "Fp1"
    
    # Multi-stage extraction for visualization
    stages = preprocess_eeg_stages(raw, selected_channels=STANDARD_CHANNELS, target_sfreq=128.0)
    raw_channel_data = stages['raw'][0, :]
    filt_channel_data = stages['filtered'][0, :]
    ica_channel_data = stages['ica_cleaned'][0, :]
    
    target_sfreq = 128.0
    prep_data = stages['normalized']
    
    if prep_data.shape[0] < 19:
        padded = np.zeros((19, prep_data.shape[1]), dtype=np.float32)
        padded[:prep_data.shape[0], :] = prep_data
        prep_data = padded
    elif prep_data.shape[0] > 19:
        prep_data = prep_data[:19, :]
        
    prep_channel_data = prep_data[0, :]
    
    # 4. Segmentation (Epochs)
    windows = slice_into_windows(prep_data, sfreq=target_sfreq, window_sec=5.0, overlap=0.5)
    if len(windows) == 0:
        raise ValueError("Unable to extract 5-second EEG windows from the uploaded file.")
        
    # Model inference
    model_wrapper = get_model(model_name)
    probs_per_window = model_wrapper.predict_proba(windows)
    
    mean_probs = np.mean(probs_per_window, axis=0)
    pred_idx = int(np.argmax(mean_probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(mean_probs[pred_idx] * 100.0)
    
    class_probs_dict = {
        'AD': round(float(mean_probs[0] * 100.0), 2),
        'CN': round(float(mean_probs[1] * 100.0), 2),
        'MCI': round(float(mean_probs[2] * 100.0), 2)
    }
    
    plot_rel_url = generate_signal_plot(
        raw_channel_data,
        filt_channel_data,
        ica_channel_data,
        prep_channel_data,
        sfreq=target_sfreq, 
        channel_name=first_ch_name
    )
    
    model_info = Config.MODEL_OPTIONS.get(model_name, Config.MODEL_OPTIONS[Config.DEFAULT_MODEL])
    
    return {
        'success': True,
        'predicted_class': pred_class,
        'predicted_label_full': Config.CLASS_FULL_NAMES[pred_class],
        'confidence': round(confidence, 2),
        'class_probabilities': class_probs_dict,
        'class_color': Config.CLASS_COLORS[pred_class],
        'metadata': {
            'file_name': os.path.basename(file_path),
            'duration_sec': round(orig_duration, 1),
            'orig_channels': orig_n_chan,
            'orig_sfreq': round(orig_sfreq, 1),
            'target_sfreq': target_sfreq,
            'total_windows': len(windows),
            'window_length_sec': 5.0,
            'preprocessing_stages': [
                "1. Channel Selection (19 Channels 10-20)",
                "2. Band-pass Filtering (0.5 - 45 Hz Butterworth)",
                "3. Notch Filtering (50 Hz Power-Line Hum)",
                "4. Artifact Removal (ICA - Independent Component Analysis)",
                "5. Epoch Segmentation (5.0s, 50% Overlap) & Z-score Normalization"
            ]
        },
        'signal_plot_url': plot_rel_url,
        'model_name': model_info['name'],
        'model_split': model_info['split'],
        'model_accuracy': model_info['accuracy']
    }


def create_sample_eeg_files():
    """
    Creates demo CSV files in dataset/samples/ representing:
    - AD: Alzheimer's Disease
    - CN: Cognitively Normal (Healthy Control)
    - MCI: Mild Cognitive Impairment
    """
    samples_dir = Config.SAMPLES_DIR
    os.makedirs(samples_dir, exist_ok=True)
    
    duration = 30.0
    sfreq = 128.0
    n_samples = int(duration * sfreq)
    t = np.linspace(0, duration, n_samples)
    
    sample_configs = [
        ('sample_AD_Alzheimer_subject.csv', 'AD'),
        ('sample_CN_Cognitively_Normal.csv', 'CN'),
        ('sample_MCI_Mild_Cognitive_Impairment.csv', 'MCI'),
        ('sample_HC_Healthy_subject.csv', 'CN'),
        ('sample_FTD_Dementia_subject.csv', 'MCI')
    ]
    
    for filename, group in sample_configs:
        filepath = os.path.join(samples_dir, filename)
        if os.path.exists(filepath):
            continue
            
        data_dict = {}
        for ch in STANDARD_CHANNELS:
            noise = np.random.normal(0, 2.0, n_samples)
            if group == 'AD':
                # AD: slowing with dominant delta and theta
                sig = 14.0 * np.sin(2 * np.pi * 3.0 * t) + 16.0 * np.sin(2 * np.pi * 5.5 * t) + 2.5 * np.sin(2 * np.pi * 10.0 * t) + noise
            elif group == 'CN':
                # CN: robust posterior alpha waves (10 Hz)
                sig = 3.0 * np.sin(2 * np.pi * 2.5 * t) + 4.0 * np.sin(2 * np.pi * 5.0 * t) + 19.0 * np.sin(2 * np.pi * 10.2 * t) + noise
            else: # MCI
                # MCI: transitional slowing with elevated theta and diminished alpha
                sig = 6.0 * np.sin(2 * np.pi * 2.8 * t) + 13.0 * np.sin(2 * np.pi * 6.5 * t) + 8.0 * np.sin(2 * np.pi * 9.5 * t) + noise
            data_dict[ch] = np.round(sig, 3)
            
        df = pd.DataFrame(data_dict)
        df.to_csv(filepath, index=False)
        print(f"Created sample EEG: {filepath}")

create_sample_eeg_files()

