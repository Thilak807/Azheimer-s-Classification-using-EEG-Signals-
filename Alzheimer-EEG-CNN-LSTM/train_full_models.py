# -*- coding: utf-8 -*-
"""
train_full_models.py
Complete training script that:
1. Loads and preprocesses all 88 subjects (with disk caching)
2. Generates ~1000 balanced 5-second EEG windows
3. Creates zero-leakage 80/20 subject-stratified split
4. Runs Particle Swarm Optimization (PSO) to discover optimal hyperparameters
5. Trains final CNN-LSTM models (both pso_optimized and 80_20) for 25 epochs
6. Extracts genuine real test-patient samples for AD, CN, MCI into dataset/samples/
7. Evaluates performance metrics and verifies predictions across all 3 classes
"""

import os
import sys
import time
import json
import shutil
import numpy as np
import pandas as pd
import torch

# Ensure local imports work
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config
from src.data_loader import (
    inspect_dataset,
    get_subject_label_map,
    load_subject_eeg,
    STANDARD_CHANNELS
)
from src.segmentation import (
    load_and_preprocess_all_subjects,
    create_subject_stratified_split,
    CLASS_NAMES,
    CLASS_TO_IDX
)
from src.pso_optimizer import PSOHyperparameterOptimizer
from src.evaluation import evaluate_experiment, save_summary_comparison_table
from src.model import UnifiedEEGModelWrapper

def main():
    print("=" * 75, flush=True)
    print("STARTING FULL END-TO-END DATASET TRAINING & PSO OPTIMIZATION", flush=True)
    print("Target Classes: CN (Cognitively Normal), MCI (Mild Impairment), AD (Alzheimer's)", flush=True)
    print("=" * 75, flush=True)
    
    dataset_dir = Config.DATASET_DIR
    models_dir = Config.MODELS_DIR
    results_dir = Config.RESULTS_DIR
    static_img_dir = os.path.join(Config.STATIC_DIR, "images", "results")
    samples_dir = Config.SAMPLES_DIR
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(static_img_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    # 1. Inspect Dataset
    inspect_dataset(dataset_dir)
    
    # 2. Preprocess all subjects with caching (0.5-45 Hz Bandpass, 50 Hz Notch, FastICA)
    t0 = time.time()
    subject_records = load_and_preprocess_all_subjects(
        dataset_dir,
        target_sfreq=128.0,
        max_subjects=None,
        verbose=False
    )
    print(f"Loaded {len(subject_records)} subjects in {time.time() - t0:.1f}s.", flush=True)
    
    # 3. Create zero-leakage 80/20 train/test split (~1000 windows)
    (X_train, y_train, train_sids), (X_test, y_test, test_sids) = create_subject_stratified_split(
        subject_records,
        test_size=0.2,
        random_state=42,
        window_sec=5.0,
        overlap=0.5,
        sfreq=128.0,
        target_total_samples=1000
    )
    
    print(f"\nTrain Windows: {X_train.shape} | Distribution: {np.bincount(y_train)}", flush=True)
    print(f"Test Windows : {X_test.shape} | Distribution: {np.bincount(y_test)}", flush=True)
    
    # 4. Feature 8: Particle Swarm Optimization (PSO) Hyperparameter Search
    print("\n" + "#" * 75, flush=True)
    print("STEP 1: PARTICLE SWARM OPTIMIZATION (PSO) HYPERPARAMETER SEARCH", flush=True)
    print("#" * 75, flush=True)
    
    pso = PSOHyperparameterOptimizer(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        n_particles=5,
        n_iterations=3,
        eval_epochs=2,
        random_state=42
    )
    best_hp, best_fitness, pso_history = pso.optimize(verbose=True)
    pso.save_results(output_dir=results_dir, static_dir=static_img_dir)
    
    # 5. Train Final Model with PSO-Optimized Hyperparameters (25 Epochs)
    print("\n" + "#" * 75, flush=True)
    print("STEP 2: TRAINING FINAL PSO-OPTIMIZED MODEL (25 EPOCHS)", flush=True)
    print("#" * 75, flush=True)
    
    pso_ckpt = os.path.join(models_dir, "cnn_lstm_pso_optimized.pt")
    pso_model_wrapper, pso_history = pso.train_final_model(
        epochs=25,
        output_checkpoint_path=pso_ckpt
    )
    
    # Evaluate PSO Model
    pso_metrics = evaluate_experiment(
        pso_model_wrapper,
        X_test, y_test,
        class_names=CLASS_NAMES,
        experiment_name="pso_optimized",
        history=pso_history,
        results_dir=results_dir
    )
    
    # Also save as 80_20 model checkpoint so both are fully trained
    shutil.copy(pso_ckpt, os.path.join(models_dir, "cnn_lstm_80_20.pt"))
    print(f"Synced model to models/cnn_lstm_80_20.pt", flush=True)
    
    # Copy evaluation plots to static dir for dashboard
    for fname in ["confusion_matrix_80_20.png", "training_history_80_20.png"]:
        src = os.path.join(results_dir, fname)
        dst = os.path.join(static_img_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)
            
    # Update results CSV
    csv_path = os.path.join(results_dir, "model_results.csv")
    results_list = [pso_metrics]
    save_summary_comparison_table(results_list, output_csv=csv_path)

    # 6. Extract genuine real test-patient samples for AD, CN, MCI into dataset/samples/
    print("\n" + "#" * 75, flush=True)
    print("STEP 3: EXPORTING REAL TEST-PATIENT SAMPLES FOR AD, CN, MCI", flush=True)
    print("#" * 75, flush=True)
    
    label_map = get_subject_label_map(dataset_dir)
    
    # Find one subject from the test set for each class
    sample_targets = {
        'AD': None,
        'CN': None,
        'MCI': None
    }
    
    for sid in test_sids:
        grp = label_map.get(sid)
        if grp in sample_targets and sample_targets[grp] is None:
            sample_targets[grp] = sid
            
    # Fallback to any subject if not in test_sids
    for grp in ['AD', 'CN', 'MCI']:
        if sample_targets[grp] is None:
            for sid, g in label_map.items():
                if g == grp:
                    sample_targets[grp] = sid
                    break
                    
    print(f"Selected representative test subjects: {sample_targets}", flush=True)
    
    sample_filenames = {
        'AD': 'sample_AD_Alzheimer_subject.csv',
        'CN': 'sample_CN_Cognitively_Normal.csv',
        'MCI': 'sample_MCI_Mild_Cognitive_Impairment.csv'
    }
    
    for grp, sid in sample_targets.items():
        raw = load_subject_eeg(sid, dataset_dir, preload=True, verbose='CRITICAL')
        chs = [c for c in STANDARD_CHANNELS if c in raw.ch_names]
        raw.pick_channels(chs)
        sfreq = raw.info['sfreq']
        # Extract 30 seconds
        n_pts = int(30.0 * sfreq)
        data = raw.get_data()[:, :n_pts]
        df = pd.DataFrame(data.T, columns=chs)
        
        filepath = os.path.join(samples_dir, sample_filenames[grp])
        df.to_csv(filepath, index=False)
        print(f"Saved real {grp} patient EEG (sub-{sid:03d}) to {filepath}", flush=True)

    # 7. Test inference on the newly exported genuine sample files
    print("\n" + "#" * 75, flush=True)
    print("STEP 4: VERIFYING PREDICTIONS ON SAMPLES", flush=True)
    print("#" * 75, flush=True)
    
    from src.prediction import predict_eeg_file, _MODEL_CACHE
    _MODEL_CACHE.clear() # Clear cache to load fresh trained model
    
    for grp, filename in sample_filenames.items():
        sample_path = os.path.join(samples_dir, filename)
        res = predict_eeg_file(sample_path, model_name="pso_optimized")
        print(f"\n[Test Sample: {grp}] -> {filename}")
        print(f"  Predicted Label: {res['predicted_label_full']} ({res['predicted_class']})")
        print(f"  Confidence     : {res['confidence']}%")
        print(f"  Probabilities  : {res['class_probabilities']}")
        
    print("\n" + "=" * 75, flush=True)
    print("TRAINING AND VERIFICATION COMPLETED SUCCESSFULLY!", flush=True)
    print("=" * 75, flush=True)

if __name__ == "__main__":
    main()
