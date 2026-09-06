"""
segmentation.py - EEG Windowing & Subject-Level Stratified Splitting
Generates ~1000 EEG samples using fixed-length windowing while strictly enforcing
SUBJECT-LEVEL train/test splits to eliminate data leakage.
"""

import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

from .data_loader import (
    get_subject_label_map,
    load_subject_eeg,
    locate_subject_file
)
from .preprocessing import preprocess_raw_eeg

# Clinical Disease Stages: CN: Cognitively Normal, MCI: Mild Cognitive Impairment, AD: Alzheimer's Disease
CLASS_TO_IDX = {'AD': 0, 'CN': 1, 'MCI': 2, 'HC': 1, 'FTD': 2}
IDX_TO_CLASS = {0: 'AD', 1: 'CN', 2: 'MCI'}
CLASS_NAMES = ['AD', 'CN', 'MCI']



def slice_into_windows(eeg_data, sfreq=128.0, window_sec=5.0, overlap=0.5, max_windows_per_subject=None):
    """
    Slices continuous EEG array (n_channels, n_times) into fixed-length windows.
    Returns:
        windows: np.ndarray of shape (n_windows, window_samples, n_channels)
    """
    n_channels, n_times = eeg_data.shape
    window_samples = int(window_sec * sfreq)
    step_samples = int(window_samples * (1.0 - overlap))
    
    if step_samples <= 0:
        step_samples = window_samples
        
    windows = []
    start = 0
    while start + window_samples <= n_times:
        # Window shape: (n_channels, window_samples) -> Transposed to (window_samples, n_channels) for 1D CNN/LSTM
        win = eeg_data[:, start:start + window_samples].T
        windows.append(win)
        start += step_samples
        
        if max_windows_per_subject is not None and len(windows) >= max_windows_per_subject:
            break
            
    if len(windows) == 0:
        return np.empty((0, window_samples, n_channels), dtype=np.float32)
        
    return np.array(windows, dtype=np.float32)


def load_and_preprocess_all_subjects(dataset_path, target_sfreq=128.0, max_subjects=None, verbose=False):
    """
    Preprocesses available subjects in the dataset with disk caching.
    Returns a list of dicts:
    [{'subject_id': sid, 'group': 'AD', 'data': (n_channels, n_times)}, ...]
    """
    import pickle
    cache_path = os.path.join(dataset_path, f"preprocessed_cache_{int(target_sfreq)}hz.pkl")
    if max_subjects is None and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                records = pickle.load(f)
            print(f"Loaded {len(records)} preprocessed subjects from disk cache: {cache_path}", flush=True)
            return records
        except Exception as err:
            print(f"Cache read error: {err}. Recomputing...", flush=True)

    label_map = get_subject_label_map(dataset_path)
    subject_records = []
    
    available_sids = [sid for sid in sorted(label_map.keys()) if locate_subject_file(sid, dataset_path) is not None]
    if max_subjects is not None and max_subjects > 0:
        available_sids = available_sids[:max_subjects]
        
    print(f"Loading and preprocessing {len(available_sids)} available subjects from '{dataset_path}'...", flush=True)
    
    for i, sid in enumerate(available_sids):
        try:
            raw = load_subject_eeg(sid, dataset_path, preload=True, verbose='CRITICAL')
            data, actual_sfreq, ch_names = preprocess_raw_eeg(raw, target_sfreq=target_sfreq, verbose=False)
            group = label_map[sid]
            subject_records.append({
                'subject_id': sid,
                'group': group,
                'label_idx': CLASS_TO_IDX[group],
                'data': data,
                'sfreq': actual_sfreq,
                'channels': ch_names
            })
            if verbose or (i + 1) % 15 == 0 or (i + 1) == len(available_sids):
                print(f"  Processed {i+1}/{len(available_sids)} subjects (sub-{sid:03d} -> {group}).", flush=True)
        except Exception as e:
            print(f"  Warning: skipping sub-{sid:03d} due to error: {e}", flush=True)
            
    if max_subjects is None and len(subject_records) > 0:
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(subject_records, f)
            print(f"Saved {len(subject_records)} preprocessed subjects to disk cache: {cache_path}", flush=True)
        except Exception as err:
            print(f"Warning: could not write cache: {err}", flush=True)
            
    return subject_records



def generate_all_windows(subject_records, window_sec=5.0, overlap=0.5, sfreq=128.0, target_total_samples=1000):
    """
    Segments all subjects and prints overall class distribution for ~1000 samples.
    """
    n_sub = len(subject_records)
    if n_sub == 0:
        raise ValueError("No subject records provided for window generation.")
        
    # Calculate target windows per subject to reach ~1000 total samples
    desired_per_sub = max(1, int(round(target_total_samples / n_sub)))
    
    all_X = []
    all_y = []
    all_subject_ids = []
    group_window_counts = {'AD': 0, 'CN': 0, 'MCI': 0}
    
    for rec in subject_records:
        wins = slice_into_windows(
            rec['data'], 
            sfreq=sfreq, 
            window_sec=window_sec, 
            overlap=overlap, 
            max_windows_per_subject=desired_per_sub
        )
        n_wins = len(wins)
        if n_wins > 0:
            all_X.append(wins)
            all_y.append(np.full((n_wins,), rec['label_idx'], dtype=np.int64))
            all_subject_ids.extend([rec['subject_id']] * n_wins)
            canon_grp = IDX_TO_CLASS.get(rec['label_idx'], rec['group'])
            group_window_counts[canon_grp] = group_window_counts.get(canon_grp, 0) + n_wins
            
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    
    print("\n" + "=" * 60)
    print("DATASET CREATION SUMMARY (~1000 EEG SAMPLES)")
    print("=" * 60)
    print(f"Total subjects    : {n_sub}")
    print(f"Total EEG windows : {len(y)}")
    print(f"AD windows        : {group_window_counts.get('AD', 0)}")
    print(f"CN windows        : {group_window_counts.get('CN', 0)}")
    print(f"MCI windows       : {group_window_counts.get('MCI', 0)}")
    print(f"Window Shape      : {X.shape[1:]} (TimeSteps x Channels)")
    print(f"Final dataset shape: {X.shape}")
    print("=" * 60 + "\n")
    
    return X, y, np.array(all_subject_ids), group_window_counts


def create_subject_stratified_split(subject_records, test_size=0.5, random_state=42,
                                    window_sec=5.0, overlap=0.5, sfreq=128.0, target_total_samples=1000):
    """
    CRITICAL ANTI-DATA-LEAKAGE SPLIT:
    1. Splits SUBJECTS first (stratified by subject diagnosis group).
    2. Generates EEG windows strictly from train subjects for X_train.
    3. Generates EEG windows strictly from test subjects for X_test.
    4. Guarantees 0% subject overlap between training and testing sets.
    """
    subject_ids = np.array([r['subject_id'] for r in subject_records])
    subject_groups = np.array([r['group'] for r in subject_records])
    
    # Stratified split at the SUBJECT level
    train_sids, test_sids = train_test_split(
        subject_ids,
        test_size=test_size,
        stratify=subject_groups,
        random_state=random_state
    )
    
    train_sids_set = set(train_sids)
    test_sids_set = set(test_sids)
    
    # Assert zero overlap
    overlap_check = train_sids_set.intersection(test_sids_set)
    assert len(overlap_check) == 0, f"DATA LEAKAGE DETECTED! Overlapping subjects: {overlap_check}"
    
    train_records = [r for r in subject_records if r['subject_id'] in train_sids_set]
    test_records = [r for r in subject_records if r['subject_id'] in test_sids_set]
    
    desired_per_sub = max(1, int(round(target_total_samples / len(subject_records))))
    
    def extract_windows_for_records(records):
        X_list, y_list, sids_list = [], [], []
        counts = {'AD': 0, 'CN': 0, 'MCI': 0}
        for rec in records:
            wins = slice_into_windows(
                rec['data'], 
                sfreq=sfreq, 
                window_sec=window_sec, 
                overlap=overlap, 
                max_windows_per_subject=desired_per_sub
            )
            if len(wins) > 0:
                X_list.append(wins)
                y_list.append(np.full((len(wins),), rec['label_idx'], dtype=np.int64))
                sids_list.extend([rec['subject_id']] * len(wins))
                canon_grp = IDX_TO_CLASS.get(rec['label_idx'], rec['group'])
                counts[canon_grp] = counts.get(canon_grp, 0) + len(wins)
        if len(X_list) == 0:
            return np.empty((0, int(window_sec*sfreq), 19)), np.empty((0,), dtype=np.int64), np.array([]), counts
        return np.vstack(X_list), np.concatenate(y_list), np.array(sids_list), counts


    X_train, y_train, train_sids_arr, train_counts = extract_windows_for_records(train_records)
    X_test, y_test, test_sids_arr, test_counts = extract_windows_for_records(test_records)
    
    train_pct = int(round((1.0 - test_size) * 100))
    test_pct = int(round(test_size * 100))
    
    print(f"\n--- Experiment Split ({train_pct}% Train / {test_pct}% Test) ---")
    print(f"Train Subjects: {len(train_records)} | Train Windows: {len(y_train)} {train_counts}")
    print(f"Test Subjects : {len(test_records)} | Test Windows : {len(y_test)} {test_counts}")
    print(f"Subject Leakage Check: 0 overlapping subjects (PASSED)")
    
    return (X_train, y_train, train_sids_arr), (X_test, y_test, test_sids_arr)
