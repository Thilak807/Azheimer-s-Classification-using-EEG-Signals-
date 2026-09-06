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


def apply_ica_artifact_removal(raw, n_components=None, max_iter=200, random_state=42, verbose=False):
    """
    3. ARTIFACT REMOVAL (ICA):
    Applies Independent Component Analysis (ICA) to identify and remove ocular (EOG/blink)
    and high-amplitude muscle artifacts from multichannel EEG.
    """
    import warnings
    n_ch = len(raw.ch_names)
    if n_components is None:
        n_components = min(15, max(2, n_ch - 1))
    else:
        n_components = min(n_components, n_ch)
        
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ica = mne.preprocessing.ICA(
                n_components=n_components,
                random_state=random_state,
                max_iter=max_iter,
                method='fastica',
                verbose=verbose
            )
            ica.fit(raw, verbose=verbose)
            
            exclude_indices = []
            # Check ocular correlation with frontal channels (Fp1, Fp2, F7, F8)
            for frontal_ch in ['Fp1', 'Fp2', 'F7', 'F8', 'FP1', 'FP2']:
                if frontal_ch in raw.ch_names:
                    try:
                        eog_inds, _ = ica.find_bads_eog(raw, ch_name=frontal_ch, threshold=2.5, verbose=verbose)
                        if eog_inds:
                            exclude_indices.extend(eog_inds)
                            break
                    except Exception:
                        pass
                        
            # Check kurtosis for spike/muscle artifacts
            try:
                sources = ica.get_sources(raw).get_data()
                from scipy.stats import kurtosis
                kurt = kurtosis(sources, axis=1)
                high_kurt_inds = [int(i) for i, k in enumerate(kurt) if abs(k) > 5.0]
                exclude_indices.extend(high_kurt_inds)
            except Exception:
                pass
                
            exclude_indices = sorted(list(set(exclude_indices)))
            max_exclude = max(1, n_components // 4)
            ica.exclude = exclude_indices[:max_exclude]
            
            raw_cleaned = ica.apply(raw.copy(), verbose=verbose)
            return raw_cleaned, len(ica.exclude)
    except Exception as e:
        return _fallback_fastica_artifact_removal(raw, n_components=n_components)



def _fallback_fastica_artifact_removal(raw, n_components=None):
    """
    Robust fallback ICA artifact removal using scikit-learn FastICA directly on data matrix.
    """
    try:
        from sklearn.decomposition import FastICA
        from scipy.stats import kurtosis
        
        data = raw.get_data()
        n_ch, n_times = data.shape
        if n_components is None:
            n_components = min(15, max(2, n_ch - 1))
            
        ica = FastICA(n_components=n_components, random_state=42, max_iter=200)
        sources = ica.fit_transform(data.T) # (n_times, n_components)
        mixing = ica.mixing_ # (n_ch, n_components)
        
        kurt = kurtosis(sources, axis=0)
        bad_comps = [int(i) for i, k in enumerate(kurt) if abs(k) > 5.0]
        max_exclude = max(1, n_components // 4)
        bad_comps = bad_comps[:max_exclude]
        
        for bc in bad_comps:
            sources[:, bc] = 0.0
            
        cleaned = np.dot(sources, mixing.T).T
        raw_cleaned = raw.copy()
        raw_cleaned._data = cleaned
        return raw_cleaned, len(bad_comps)
    except Exception:
        return raw.copy(), 0


def preprocess_raw_eeg(raw, 
                       selected_channels=STANDARD_CHANNELS, 
                       target_sfreq=128.0, 
                       l_freq=0.5, 
                       h_freq=45.0, 
                       notch_freq=50.0,
                       apply_ica=True,
                       verbose=False):
    """
    Complete 5-Stage EEG Preprocessing Pipeline (Box 2 in Architecture Diagram):
    1. Channel Selection (10-20 International System, 19 channels)
    2. Band-pass Filtering (0.5 - 45 Hz Butterworth filter)
    3. Notch Filtering (50 Hz / 60 Hz power-line noise elimination)
    4. Artifact Removal via ICA (Independent Component Analysis)
    5. Resampling (128 Hz) & Z-Score Normalization per channel
    
    Returns:
        preprocessed_data: np.ndarray of shape (n_channels, n_times)
        actual_sfreq: float
        channel_names: list of str
    """
    raw_copy = raw.copy()
    
    # 1. Channel Selection
    available_channels = [ch for ch in selected_channels if ch in raw_copy.ch_names]
    if len(available_channels) == 0:
        available_channels = raw_copy.ch_names[:19]
    raw_copy.pick_channels(available_channels)
    
    raw_shape = raw_copy.get_data().shape
    if verbose:
        print(f"  [Preprocess Stage 1] Selected Channels: {len(available_channels)}, Samples: {raw_shape[1]}")
    
    # 2. Band-pass Filter (0.5 - 45 Hz)
    raw_copy.filter(l_freq=l_freq, h_freq=h_freq, method='iir', verbose=False)
    if verbose:
        print(f"  [Preprocess Stage 2] Band-pass filtered ({l_freq}-{h_freq} Hz)")
        
    # 3. Notch Filter (50 Hz power line noise)
    if notch_freq is not None and notch_freq > 0:
        if raw_copy.info['sfreq'] > notch_freq * 2:
            raw_copy.notch_filter(freqs=notch_freq, method='iir', verbose=False)
            if verbose:
                print(f"  [Preprocess Stage 3] Notch filtered ({notch_freq} Hz)")
                
    # 4. Artifact Removal (ICA - Independent Component Analysis)
    n_artifacts_removed = 0
    if apply_ica:
        raw_copy, n_artifacts_removed = apply_ica_artifact_removal(raw_copy, verbose=verbose)
        if verbose:
            print(f"  [Preprocess Stage 4] ICA Artifacts Removed: {n_artifacts_removed} components")
            
    # Resampling to target_sfreq (128 Hz)
    if target_sfreq is not None and target_sfreq > 0 and raw_copy.info['sfreq'] != target_sfreq:
        raw_copy.resample(sfreq=target_sfreq, verbose=False)
        if verbose:
            print(f"  [Preprocess Resample] Resampled to {target_sfreq} Hz")
            
    # 5. Extract numpy array & Normalize (Z-score per channel)
    data = raw_copy.get_data() # shape: (n_channels, n_times)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Artifact clipping (clip extreme spikes > 5 std) and Z-score normalization
    for ch_idx in range(data.shape[0]):
        std = np.std(data[ch_idx, :])
        mean = np.mean(data[ch_idx, :])
        if std > 0:
            data[ch_idx, :] = np.clip(data[ch_idx, :], mean - 5*std, mean + 5*std)
            data[ch_idx, :] = (data[ch_idx, :] - mean) / std
        else:
            data[ch_idx, :] = data[ch_idx, :] - mean
            
    return data, raw_copy.info['sfreq'], available_channels


def preprocess_eeg_stages(raw, selected_channels=STANDARD_CHANNELS, target_sfreq=128.0):
    """
    Extracts intermediate signal arrays across all 5 stages of the preprocessing pipeline:
    Returns dict:
      'raw': original signal array
      'filtered': band-pass + notch filtered signal
      'ica_cleaned': signal after ICA artifact removal
      'normalized': final z-score normalized signal
      'channels': channel names
      'sfreq': target sfreq
    """
    raw_copy = raw.copy()
    available = [ch for ch in selected_channels if ch in raw_copy.ch_names]
    if len(available) == 0:
        available = raw_copy.ch_names[:19]
    raw_copy.pick_channels(available)
    raw_sig = raw_copy.get_data().copy()
    
    # Filtered
    raw_copy.filter(l_freq=0.5, h_freq=45.0, method='iir', verbose=False)
    if raw_copy.info['sfreq'] > 100.0:
        raw_copy.notch_filter(freqs=50.0, method='iir', verbose=False)
    filtered_sig = raw_copy.get_data().copy()
    
    # ICA Cleaned
    raw_ica, _ = apply_ica_artifact_removal(raw_copy, verbose=False)
    if target_sfreq is not None and target_sfreq > 0 and raw_ica.info['sfreq'] != target_sfreq:
        raw_ica.resample(sfreq=target_sfreq, verbose=False)
    ica_sig = raw_ica.get_data().copy()
    
    # Normalized
    norm_sig = np.nan_to_num(ica_sig.copy(), nan=0.0, posinf=0.0, neginf=0.0)
    for ch_idx in range(norm_sig.shape[0]):
        std = np.std(norm_sig[ch_idx, :])
        mean = np.mean(norm_sig[ch_idx, :])
        if std > 0:
            norm_sig[ch_idx, :] = np.clip(norm_sig[ch_idx, :], mean - 5*std, mean + 5*std)
            norm_sig[ch_idx, :] = (norm_sig[ch_idx, :] - mean) / std
            
    return {
        'raw': raw_sig,
        'filtered': filtered_sig,
        'ica_cleaned': ica_sig,
        'normalized': norm_sig,
        'channels': available,
        'sfreq': target_sfreq
    }



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
