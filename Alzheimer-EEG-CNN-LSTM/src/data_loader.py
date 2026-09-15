"""
data_loader.py - Dataset Loader for EEG Dataset
Handles subject discovery, MATLAB .mat multi-segment EEG parsing, MNE Raw loading,
and dataset inspection for Alzheimer's (AD), Cognitively Normal (CN), and Mild Cognitive Impairment (MCI).
Supports both the real MATLAB clinical dataset and BIDS/EEGLAB formats.
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import scipy.io as sio
import mne

# Standard 19 EEG Channels (10-20 International System)
STANDARD_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
]

GROUP_RANGES = {
    'AD': range(1, 37),    # Default fallback
    'CN': range(37, 66),
    'MCI': range(66, 89)
}


def find_dataset_root(candidate_paths=None):
    """
    Automatically search for the dataset root directory across standard paths.
    """
    if candidate_paths is None:
        candidate_paths = [
            os.path.join(os.path.dirname(__file__), "..", "dataset"),
            os.path.join(os.getcwd(), "dataset"),
            os.path.join(os.getcwd(), "Alzheimer-EEG-CNN-LSTM", "dataset"),
            "/kaggle/input/openneuro-ds004504/ds004504",
            "dataset",
            "../dataset"
        ]
    elif isinstance(candidate_paths, str):
        candidate_paths = [candidate_paths, "dataset"]

    for p in candidate_paths:
        if p and os.path.exists(p):
            if (os.path.exists(os.path.join(p, "AD")) and os.path.exists(os.path.join(p, "CONTROL"))) or \
               os.path.exists(os.path.join(p, "participants.tsv")) or \
               os.path.exists(os.path.join(p, "derivatives")) or \
               len(glob.glob(os.path.join(p, "sub-*"))) > 0:
                return os.path.abspath(p)

    return os.path.abspath(candidate_paths[0] if candidate_paths else "dataset")


def discover_all_subjects(dataset_path):
    """
    Discovers all available subjects in the dataset folder.
    Supports:
      - AD folder: AD1..AD59 (multi-segment .mat files)
      - MCI folder: MCI1..MCI7 (multi-segment .mat files)
      - CONTROL folder: normal1, normal2, plus all *fir.mat files
      - BIDS / OpenNeuro format (derivatives/sub-* or sub-*)
    Returns list of dicts:
      [{'subject_id': 'AD1', 'group': 'AD', 'files': [...], 'sfreq': 256.0}, ...]
    """
    subjects = []
    ad_dir = os.path.join(dataset_path, "AD")
    mci_dir = os.path.join(dataset_path, "MCI")
    ctrl_dir = os.path.join(dataset_path, "CONTROL")

    has_mat_dataset = os.path.exists(ad_dir) or os.path.exists(ctrl_dir) or os.path.exists(mci_dir)

    if has_mat_dataset:
        # 1. AD Subjects (AD1..AD59)
        # AD1-34: 256 Hz, AD35-59: 128 Hz (per readme.txt)
        if os.path.exists(ad_dir):
            subdirs = [d for d in os.listdir(ad_dir) if os.path.isdir(os.path.join(ad_dir, d))]
            def sort_key(name):
                nums = re.findall(r'\d+', name)
                return int(nums[0]) if nums else 999
            subdirs = sorted(subdirs, key=sort_key)
            for d in subdirs:
                s_path = os.path.join(ad_dir, d)
                files = sorted(glob.glob(os.path.join(s_path, "*.mat")), key=sort_key)
                if files:
                    num_match = re.findall(r'\d+', d)
                    sub_num = int(num_match[0]) if num_match else 1
                    sfreq = 256.0 if sub_num <= 34 else 128.0
                    subjects.append({
                        'subject_id': d,
                        'group': 'AD',
                        'files': files,
                        'sfreq': sfreq
                    })

        # 2. MCI Subjects (MCI1..MCI7)
        # MCI1-4: 256 Hz, MCI5-7: 128 Hz (per readme.txt)
        if os.path.exists(mci_dir):
            subdirs = [d for d in os.listdir(mci_dir) if os.path.isdir(os.path.join(mci_dir, d))]
            subdirs = sorted(subdirs, key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 999)
            for d in subdirs:
                s_path = os.path.join(mci_dir, d)
                files = sorted(glob.glob(os.path.join(s_path, "*.mat")), key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]) if re.findall(r'\d+', os.path.basename(x)) else 999)
                if files:
                    num_match = re.findall(r'\d+', d)
                    sub_num = int(num_match[0]) if num_match else 1
                    sfreq = 256.0 if sub_num <= 4 else 128.0
                    subjects.append({
                        'subject_id': d,
                        'group': 'MCI',
                        'files': files,
                        'sfreq': sfreq
                    })

        # 3. CONTROL Subjects (normal1, normal2, and *fir.mat files)
        # All 128 Hz
        if os.path.exists(ctrl_dir):
            subdirs = [d for d in os.listdir(ctrl_dir) if os.path.isdir(os.path.join(ctrl_dir, d))]
            for d in sorted(subdirs):
                s_path = os.path.join(ctrl_dir, d)
                files = sorted(glob.glob(os.path.join(s_path, "*.mat")), key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]) if re.findall(r'\d+', os.path.basename(x)) else 999)
                if files:
                    subjects.append({
                        'subject_id': f"CN_{d}",
                        'group': 'CN',
                        'files': files,
                        'sfreq': 128.0
                    })
            # Individual *fir.mat files in CONTROL
            fir_files = sorted(glob.glob(os.path.join(ctrl_dir, "*fir.mat")))
            for f in fir_files:
                base = os.path.basename(f).replace('.mat', '')
                subjects.append({
                    'subject_id': f"CN_{base}",
                    'group': 'CN',
                    'files': [f],
                    'sfreq': 128.0
                })

    if not subjects:
        # Fallback to BIDS / EEGLAB .set format
        tsv_path = os.path.join(dataset_path, "participants.tsv")
        if os.path.exists(tsv_path):
            df = pd.read_csv(tsv_path, sep="\t")
            for _, row in df.iterrows():
                sid_str = str(row.get('participant_id', ''))
                grp = str(row.get('Group', 'CN')).strip()
                if grp in ['A', 'Alzheimer']:
                    grp = 'AD'
                elif grp in ['C', 'HC', 'Control', 'Healthy']:
                    grp = 'CN'
                elif grp in ['F', 'FTD']:
                    grp = 'MCI'
                sid_clean = sid_str.replace('sub-', '')
                sid_num = int(sid_clean) if sid_clean.isdigit() else 1
                fpath = locate_subject_file(sid_num, dataset_path)
                if fpath:
                    subjects.append({
                        'subject_id': sid_str if sid_str else f"sub-{sid_num:03d}",
                        'group': grp,
                        'files': [fpath],
                        'sfreq': 128.0
                    })

    return subjects


def load_participants_metadata(dataset_path):
    """
    Loads or generates participants metadata DataFrame.
    """
    subjects = discover_all_subjects(dataset_path)
    records = []
    for s in subjects:
        grp = s['group']
        sid = s['subject_id']
        records.append({
            'participant_id': sid,
            'subject_id': sid,
            'Group': grp,
            'SamplingRate': s['sfreq'],
            'TotalFiles': len(s['files'])
        })
    if records:
        return pd.DataFrame(records)
    return pd.DataFrame(columns=['participant_id', 'subject_id', 'Group', 'SamplingRate', 'TotalFiles'])


def get_subject_label_map(dataset_path):
    """
    Returns a dictionary mapping subject_id -> group label ('AD', 'CN', 'MCI').
    """
    subjects = discover_all_subjects(dataset_path)
    return {s['subject_id']: s['group'] for s in subjects}


def locate_subject_file(subject_id, dataset_path):
    """
    Finds the file path(s) for a given subject_id.
    """
    subjects = discover_all_subjects(dataset_path)
    for s in subjects:
        if str(s['subject_id']) == str(subject_id):
            return s['files'][0] if s['files'] else None
    return None


def _load_single_mat_array(fpath):
    """
    Loads and extracts a 2D EEG array from a MATLAB .mat file.
    Returns: np.ndarray of shape (19, N_times)
    """
    mat = sio.loadmat(fpath)
    data_arr = None
    for k in ['export', 'segmenty', 'data', 'EEG']:
        if k in mat and isinstance(mat[k], np.ndarray):
            data_arr = mat[k]
            break
    if data_arr is None:
        for k, v in mat.items():
            if isinstance(v, np.ndarray) and v.ndim == 2:
                data_arr = v
                break
    if data_arr is None:
        raise ValueError(f"No valid 2D EEG array found in {fpath}")

    # Ensure shape is (n_channels, n_times)
    if data_arr.shape[0] > data_arr.shape[1]:
        data_arr = data_arr.T

    n_ch = min(data_arr.shape[0], 19)
    arr = data_arr[:n_ch, :].astype(np.float32)

    # Pad to 19 channels if needed
    if arr.shape[0] < 19:
        pad = np.zeros((19, arr.shape[1]), dtype=np.float32)
        pad[:arr.shape[0], :] = arr
        arr = pad

    return arr


def load_subject_eeg(subject_info_or_id, dataset_path, preload=True, verbose='CRITICAL'):
    """
    Loads EEG data for a single subject using MNE RawArray / EEGLAB reader.
    Concatenates multiple recording segments if available and resamples to 128 Hz.
    Returns: mne.io.Raw instance
    """
    if isinstance(subject_info_or_id, dict):
        sub_info = subject_info_or_id
    else:
        subjects = discover_all_subjects(dataset_path)
        matching = [s for s in subjects if str(s['subject_id']) == str(subject_info_or_id)]
        if not matching:
            raise FileNotFoundError(f"Subject '{subject_info_or_id}' not found in dataset '{dataset_path}'")
        sub_info = matching[0]

    files = sub_info['files']
    orig_sfreq = float(sub_info.get('sfreq', 128.0))

    if not files:
        raise FileNotFoundError(f"No files found for subject {sub_info['subject_id']}")

    # If first file is .set (EEGLAB)
    if files[0].lower().endswith('.set'):
        raw = mne.io.read_raw_eeglab(files[0], preload=preload, verbose=verbose)
        if raw.info['sfreq'] != 128.0:
            raw.resample(128.0, verbose=verbose)
        return raw

    # Otherwise MATLAB .mat files: load and concatenate all segments
    arrays = [_load_single_mat_array(f) for f in files]
    concat_arr = np.concatenate(arrays, axis=1)

    # Convert uV to V if needed (MNE expects Volts)
    if np.max(np.abs(concat_arr)) > 1.0:
        concat_arr = concat_arr * 1e-6

    info = mne.create_info(ch_names=STANDARD_CHANNELS, sfreq=orig_sfreq, ch_types='eeg')
    raw = mne.io.RawArray(concat_arr, info, verbose=verbose)

    # Resample to 128 Hz
    if orig_sfreq != 128.0:
        raw.resample(128.0, verbose=verbose)

    return raw


def inspect_dataset(dataset_path):
    """
    Inspects available dataset files, metadata, subjects, sampling frequency, and channel count.
    """
    print("=" * 70)
    print("DATASET INSPECTION: Real Clinical EEG Dataset (AD / MCI / CN)")
    print("=" * 70)
    print(f"Dataset Root Path: {dataset_path}")

    subjects = discover_all_subjects(dataset_path)
    class_counts = {}
    total_files = 0
    for s in subjects:
        grp = s['group']
        class_counts[grp] = class_counts.get(grp, 0) + 1
        total_files += len(s['files'])

    print(f"Total Discovered Subjects : {len(subjects)}")
    print(f"Total Recording Files     : {total_files}")
    print("\nClass Distribution:")
    for grp, cnt in sorted(class_counts.items()):
        print(f"  - {grp:5s}: {cnt:3d} subjects")

    if subjects:
        try:
            sample_raw = load_subject_eeg(subjects[0], dataset_path)
            print(f"\nSample Subject: {subjects[0]['subject_id']}")
            print(f"  Sampling Frequency (after resample) : {sample_raw.info['sfreq']} Hz")
            print(f"  Channels                            : {len(sample_raw.ch_names)} ({sample_raw.ch_names[:5]}...)")
            print(f"  Total Duration                      : {sample_raw.times[-1]:.1f} seconds")
        except Exception as e:
            print(f"Warning reading sample EEG: {e}")

    print("=" * 70)
    return {
        'total_configured': len(subjects),
        'available_subjects': [s['subject_id'] for s in subjects],
        'missing_subjects': [],
        'class_distribution': class_counts,
        'subjects': subjects
    }


def generate_synthetic_openneuro_dataset(output_dir, n_subjects=88, duration_sec=120, sfreq=500.0):
    """
    Fallback dataset generator when no real dataset exists.
    """
    os.makedirs(output_dir, exist_ok=True)
    derivatives_dir = os.path.join(output_dir, "derivatives")
    os.makedirs(derivatives_dir, exist_ok=True)

    records = []
    np.random.seed(42)
    n_samples = int(duration_sec * sfreq)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    print(f"Generating synthetic EEG dataset in {output_dir} ({n_subjects} subjects)...")

    chanlocs = np.empty((len(STANDARD_CHANNELS),), dtype=[('labels', 'O')])
    for idx, ch in enumerate(STANDARD_CHANNELS):
        chanlocs[idx] = (ch,)

    for sid in range(1, n_subjects + 1):
        sid_str = f"{sid:03d}"
        if sid <= 36:
            group = 'AD'
        elif sid <= 65:
            group = 'HC'
        else:
            group = 'FTD'

        records.append({
            'participant_id': f'sub-{sid_str}',
            'Gender': 'F' if sid % 2 == 0 else 'M',
            'Age': 70,
            'Group': group,
            'MMSE': 20.0
        })

        sub_eeg_dir = os.path.join(derivatives_dir, f"sub-{sid_str}", "eeg")
        os.makedirs(sub_eeg_dir, exist_ok=True)
        set_file = os.path.join(sub_eeg_dir, f"sub-{sid_str}_task-eyesclosed_eeg.set")

        data = np.zeros((len(STANDARD_CHANNELS), n_samples), dtype=np.float32)
        for ch_idx in range(len(STANDARD_CHANNELS)):
            data[ch_idx, :] = np.sin(2 * np.pi * 10.0 * t) * 1e-6

        eeg_struct = {
            'setname': f'sub-{sid_str}_eyesclosed',
            'srate': float(sfreq),
            'nbchan': int(len(STANDARD_CHANNELS)),
            'pnts': int(n_samples),
            'trials': 1,
            'data': data,
            'chanlocs': chanlocs,
            'xmin': 0.0,
            'xmax': float(duration_sec)
        }
        sio.savemat(set_file, {'EEG': eeg_struct})

    df_tsv = pd.DataFrame(records)
    df_tsv.to_csv(os.path.join(output_dir, "participants.tsv"), sep="\t", index=False)

