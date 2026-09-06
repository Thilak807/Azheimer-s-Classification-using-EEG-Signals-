# -*- coding: utf-8 -*-
"""
evaluate_1000_samples.py - Benchmark evaluation on a 1,000-sample EEG dataset.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import find_dataset_root, generate_synthetic_openneuro_dataset
from src.segmentation import load_and_preprocess_all_subjects, generate_all_windows
from src.prediction import get_model


def run_1000_sample_test(model_name="80_20", target_samples=1000):
    print("=" * 70)
    print("      EEG HYBRID CNN-LSTM: 1,000 SAMPLE DATASET EVALUATION")
    print("=" * 70)

    dataset_path = find_dataset_root([
        os.path.join(BASE_DIR, "dataset"),
        "dataset",
        os.path.join(os.path.dirname(BASE_DIR), "dataset")
    ])
    print(f"[*] Dataset Location : {dataset_path}")

    from src.data_loader import get_subject_label_map, locate_subject_file
    label_map = get_subject_label_map(dataset_path)
    available = [sid for sid in label_map.keys() if locate_subject_file(sid, dataset_path) is not None]

    if len(available) == 0:
        print("[!] No raw EEG files found in dataset path. Generating benchmark dataset...")
        generate_synthetic_openneuro_dataset(dataset_path, n_subjects=88, duration_sec=120, sfreq=500.0)

    print("[+] Loading and preprocessing EEG subjects...")
    records = load_and_preprocess_all_subjects(dataset_path, target_sfreq=128.0, verbose=False)

    print(f"[+] Extracting ~{target_samples} window samples (5.0s, 19 channels)...")
    X_test, y_test, sids, counts = generate_all_windows(records, target_total_samples=target_samples)

    total = len(y_test)
    print("\n[+] Total Evaluated Samples:", total)
    print(f"    - AD  (Alzheimer's Disease)      : {counts.get('AD', 0)}")
    print(f"    - HC  (Healthy Control)          : {counts.get('HC', 0)}")
    print(f"    - FTD (Frontotemporal Dementia)  : {counts.get('FTD', 0)}")

    print("\n[*] Running Inference with Hybrid CNN-LSTM Model...")
    model = get_model(model_name)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    print("\n" + "=" * 70)
    print("                     EVALUATION REPORT")
    print("=" * 70)
    print(f"Overall Accuracy  : {acc * 100:.2f}% (Precision: {prec * 100:.2f}%, Recall: {rec * 100:.2f}%, F1: {f1 * 100:.2f}%)")
    print("-" * 70)
    print(classification_report(y_test, y_pred, target_names=['AD', 'HC', 'FTD'], digits=4, zero_division=0))
    print("-" * 70)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("=" * 70 + "\n")


if __name__ == '__main__':
    run_1000_sample_test()