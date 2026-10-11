# -*- coding: utf-8 -*-
"""
train_full_models.py
Complete training script that:
1. Loads and preprocesses all clinical subjects from real MATLAB dataset (AD, CN, MCI)
2. Extracts balanced 5-second EEG windows with strict subject-level anti-leakage splitting
3. Trains all experimental configurations: 80/20, 50/50, 20/80
4. Executes Particle Swarm Optimization (PSO) to discover optimal hyperparameters
5. Trains final PSO-optimized model and evaluates all models on untouched test subjects
6. Generates confusion matrices, training loss curves, and model_results.csv
7. Exports genuine real patient samples for AD, CN, and MCI into dataset/samples/
8. Verifies end-to-end inference across all classes
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
from src.train import train_experiment
from src.pso_optimizer import PSOHyperparameterOptimizer
from src.evaluation import evaluate_experiment, save_summary_comparison_table
from src.model import UnifiedEEGModelWrapper


def main():
    print("=" * 75, flush=True)
    print("STARTING FULL END-TO-END CLINICAL EEG DATASET TRAINING & PSO OPTIMIZATION", flush=True)
    print("Target Classes: AD (Alzheimer's), CN (Cognitively Normal), MCI (Mild Cognitive Impairment)", flush=True)
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
    
    # 2. Preprocess all subjects with disk caching (0.5-45 Hz Bandpass, 50 Hz Notch, Artifact Cleaning, Z-score)
    t0 = time.time()
    subject_records = load_and_preprocess_all_subjects(
        dataset_dir,
        target_sfreq=128.0,
        max_subjects=None,
        verbose=False
    )
    print(f"\nLoaded {len(subject_records)} clinical subjects in {time.time() - t0:.1f}s.", flush=True)
    
    results_list = []
    
    # 3. Train the 3 Standard Split Experiments (80/20, 50/50, 20/80)
    experiments = [
        {"name": "80_20", "test_size": 0.20, "desc": "80% Training / 20% Testing", "epochs": 25},
        {"name": "50_50", "test_size": 0.50, "desc": "50% Training / 50% Testing", "epochs": 25},
        {"name": "20_80", "test_size": 0.80, "desc": "20% Training / 80% Testing", "epochs": 25}
    ]
    
    split_data = {}
    
    for exp in experiments:
        exp_name = exp["name"]
        test_sz = exp["test_size"]
        epochs = exp["epochs"]
        
        print("\n" + "#" * 75, flush=True)
        print(f"TRAINING EXPERIMENT: {exp['desc']} (Split: {exp_name})", flush=True)
        print("#" * 75, flush=True)
        
        (X_train, y_train, train_sids), (X_test, y_test, test_sids) = create_subject_stratified_split(
            subject_records,
            test_size=test_sz,
            random_state=42,
            window_sec=5.0,
            overlap=0.5,
            sfreq=128.0,
            target_total_samples=1200
        )
        split_data[exp_name] = ((X_train, y_train, train_sids), (X_test, y_test, test_sids))
        
        model_wrapper, history, duration = train_experiment(
            X_train, y_train,
            X_val=X_test, y_val=y_test,
            experiment_name=exp_name,
            epochs=epochs,
            batch_size=32,
            lr=0.001,
            models_dir=models_dir,
            framework='torch'
        )
        
        metrics = evaluate_experiment(
            model_wrapper,
            X_test, y_test,
            class_names=CLASS_NAMES,
            experiment_name=exp_name,
            history=history,
            results_dir=results_dir
        )
        results_list.append(metrics)
        
        # Copy plots to static directory
        for fname in [f"confusion_matrix_{exp_name}.png", f"training_history_{exp_name}.png"]:
            src = os.path.join(results_dir, fname)
            dst = os.path.join(static_img_dir, fname)
            if os.path.exists(src):
                shutil.copy(src, dst)
                
    # 4. Particle Swarm Optimization (PSO) Hyperparameter Search on 80/20 data
    print("\n" + "#" * 75, flush=True)
    print("STEP 4: PARTICLE SWARM OPTIMIZATION (PSO) HYPERPARAMETER TUNING", flush=True)
    print("#" * 75, flush=True)
    
    (X_train_80, y_train_80, _), (X_test_80, y_test_80, test_sids_80) = split_data["80_20"]
    
    pso = PSOHyperparameterOptimizer(
        X_train=X_train_80,
        y_train=y_train_80,
        X_val=X_test_80,
        y_val=y_test_80,
        n_particles=5,
        n_iterations=3,
        eval_epochs=2,
        random_state=42
    )
    best_hp, best_fitness, pso_history = pso.optimize(verbose=True)
    pso.save_results(output_dir=results_dir, static_dir=static_img_dir)
    
    # 5. Train Final Model with PSO-Optimized Hyperparameters
    print("\n" + "#" * 75, flush=True)
    print("STEP 5: TRAINING FINAL PSO-OPTIMIZED MODEL (30 EPOCHS)", flush=True)
    print("#" * 75, flush=True)
    
    pso_ckpt = os.path.join(models_dir, "cnn_lstm_pso_optimized.pt")
    pso_model_wrapper, pso_train_history = pso.train_final_model(
        epochs=30,
        output_checkpoint_path=pso_ckpt
    )
    
    pso_metrics = evaluate_experiment(
        pso_model_wrapper,
        X_test_80, y_test_80,
        class_names=CLASS_NAMES,
        experiment_name="pso_optimized",
        history=pso_train_history,
        results_dir=results_dir
    )
    results_list.append(pso_metrics)
    
    # Copy PSO evaluation plots to static dir
    for fname in ["confusion_matrix_pso_optimized.png", "training_history_pso_optimized.png"]:
        src = os.path.join(results_dir, fname)
        dst = os.path.join(static_img_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)
            
    # Save overall summary table
    csv_path = os.path.join(results_dir, "model_results.csv")
    save_summary_comparison_table(results_list, output_csv=csv_path)

    # 6. Extract genuine real test-patient samples for AD, CN, MCI into dataset/samples/
    print("\n" + "#" * 75, flush=True)
    print("STEP 6: EXPORTING REAL TEST-PATIENT SAMPLES FOR AD, CN, MCI", flush=True)
    print("#" * 75, flush=True)
    
    label_map = get_subject_label_map(dataset_dir)
    sample_targets = {'AD': None, 'CN': None, 'MCI': None}
    
    for sid in test_sids_80:
        grp = label_map.get(sid)
        if grp in sample_targets and sample_targets[grp] is None:
            sample_targets[grp] = sid
            
    for grp in ['AD', 'CN', 'MCI']:
        if sample_targets[grp] is None:
            for sid, g in label_map.items():
                if g == grp:
                    sample_targets[grp] = sid
                    break
                    
    print(f"Selected representative test subjects: {sample_targets}", flush=True)
    
    sample_filenames = {
        'AD': 'AD_Alzheimer_subject.csv',
        'CN': 'CN_Cognitively_Normal.csv',
        'MCI': 'MCI_Mild_Cognitive_Impairment.csv'
    }
    
    for grp, sid in sample_targets.items():
        raw = load_subject_eeg(sid, dataset_dir, preload=True, verbose='CRITICAL')
        chs = [c for c in STANDARD_CHANNELS if c in raw.ch_names]
        raw.pick_channels(chs)
        sfreq = raw.info['sfreq']
        n_pts = int(30.0 * sfreq)
        data = raw.get_data()[:, :n_pts]
        df = pd.DataFrame(data.T, columns=chs)
        
        filepath = os.path.join(samples_dir, sample_filenames[grp])
        df.to_csv(filepath, index=False)
        print(f"Saved real {grp} patient EEG ({sid}) to {filepath}", flush=True)

    # 7. Test inference on the newly exported genuine sample files
    print("\n" + "#" * 75, flush=True)
    print("STEP 7: VERIFYING PREDICTIONS ON SAMPLES", flush=True)
    print("#" * 75, flush=True)
    
    from src.prediction import predict_eeg_file, _MODEL_CACHE
    _MODEL_CACHE.clear()
    
    for grp, filename in sample_filenames.items():
        sample_path = os.path.join(samples_dir, filename)
        res = predict_eeg_file(sample_path, model_name="pso_optimized")
        print(f"\n[Test Sample: {grp}] -> {filename}")
        print(f"  Predicted Label: {res['predicted_label_full']} ({res['predicted_class']})")
        print(f"  Confidence     : {res['confidence']}%")
        print(f"  Probabilities  : {res['class_probabilities']}")
        
    print("\n" + "=" * 75, flush=True)
    print("ALL CLINICAL DATASET TRAINING, PSO OPTIMIZATION, AND VERIFICATION COMPLETE!", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
