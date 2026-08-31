"""
preprocessing.py - EEG Preprocessing Pipeline
Provides filtering (0.5-45 Hz bandpass, 50 Hz notch), channel selection (19 channels),
downsampling (to 128 Hz), artifact handling, z-score normalization, and PSD spectral analysis.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, welch
import mne
from .data_loader import STANDARD_CHANNELS


def preprocess_raw_eeg(raw, 
                       selected_channels=STANDARD_CHANNELS, 
                       target_sfreq=128.0, 
                       l_freq=0.5, 
                       h_freq=45.0, 
                       notch_freq=50.0, 
                       verbose=False):
    """
    Complete preprocessing pipeline for an MNE Raw object:
    1. Selects specified EEG channels
    2. Applies 0.5 - 45 Hz Band-Pass filter
    3. Applies 50 Hz Notch filter
    4. Resamples to target_sfreq (128 Hz)
    5. Applies Z-score normalization per channel
    
    Returns:
        preprocessed_data: np.ndarray of shape (n_channels, n_times)
        actual_sfreq: float
        channel_names: list of str
    """
    raw_copy = raw.copy()
    
    # 1. Channel Selection
    available_channels = [ch for ch in selected_channels if ch in raw_copy.ch_names]
    if len(available_channels) == 0:
        # Fallback to whatever channels exist
        available_channels = raw_copy.ch_names[:19]
    raw_copy.pick_channels(available_channels)
    
    raw_shape = raw_copy.get_data().shape
    if verbose:
        print(f"  [Preprocess] Raw EEG Shape: {raw_shape} ({len(available_channels)} channels, {raw_shape[1]} samples)")
    
    # 2. Band-pass filter (0.5 - 45 Hz)
    raw_copy.filter(l_freq=l_freq, h_freq=h_freq, method='iir', verbose=False)
    
    # 3. Notch filter (50 Hz power line noise)
    if notch_freq is not None and notch_freq > 0:
        if raw_copy.info['sfreq'] > notch_freq * 2:
            raw_copy.notch_filter(freqs=notch_freq, method='iir', verbose=False)
            
    filtered_shape = raw_copy.get_data().shape
    if verbose:
        print(f"  [Preprocess] Filtered EEG Shape: {filtered_shape}")
        
    # 4. Resampling
    if target_sfreq is not None and target_sfreq > 0 and raw_copy.info['sfreq'] != target_sfreq:
        raw_copy.resample(sfreq=target_sfreq, verbose=False)
        
    resampled_shape = raw_copy.get_data().shape
    if verbose:
        print(f"  [Preprocess] Resampled ({target_sfreq}Hz) Shape: {resampled_shape}")
        
    # 5. Extract numpy array & Normalize (Z-score per channel)
    data = raw_copy.get_data() # shape: (n_channels, n_times)
    
    # Handle NaN or Inf values
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Artifact clipping (clip extreme spikes > 5 standard deviations)
    for ch_idx in range(data.shape[0]):
        std = np.std(data[ch_idx, :])
        mean = np.mean(data[ch_idx, :])
        if std > 0:
            data[ch_idx, :] = np.clip(data[ch_idx, :], mean - 5*std, mean + 5*std)
            # Z-score normalization
            data[ch_idx, :] = (data[ch_idx, :] - mean) / std
        else:
            data[ch_idx, :] = data[ch_idx, :] - mean
            
    return data, raw_copy.info['sfreq'], available_channels


def compute_welch_psd(data, sfreq=128.0, n_fft=512):
    """
    Computes Welch Power Spectral Density for multidimensional EEG data.
    data shape: (n_channels, n_times)
    Returns: freqs (n_freqs,), psd (n_channels, n_freqs)
    """
    freqs, psd = welch(data, fs=sfreq, nperseg=min(n_fft, data.shape[1]), axis=-1)
    return freqs, psd


def compute_band_powers(data, sfreq=128.0, bands=None):
    """
    Computes absolute band power for standard EEG rhythms:
    Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-45 Hz).
    """
    if bands is None:
        bands = {
            'Delta': (0.5, 4.0),
            'Theta': (4.0, 8.0),
            'Alpha': (8.0, 13.0),
            'Beta': (13.0, 30.0),
            'Gamma': (30.0, 45.0)
        }
        
    freqs, psd = compute_welch_psd(data, sfreq=sfreq)
    band_power_dict = {}
    
    for band_name, (low, high) in bands.items():
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        if np.any(idx_band):
            # Integrate PSD over frequency band (trapezoidal rule)
            bp = np.trapz(psd[:, idx_band], freqs[idx_band], axis=-1)
            band_power_dict[band_name] = bp
        else:
            band_power_dict[band_name] = np.zeros(data.shape[0])
            
    return band_power_dict
