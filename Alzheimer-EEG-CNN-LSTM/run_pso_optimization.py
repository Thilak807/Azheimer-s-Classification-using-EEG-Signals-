# -*- coding: utf-8 -*-
"""
run_pso_optimization.py - Standalone Particle Swarm Optimization (PSO) Runner
Optimizes Hybrid CNN-LSTM Hyperparameters (Learning Rate, CNN Filters, LSTM Units,
Dropout Rate, Batch Size) on EEG Dataset.
Box 8 in Architecture Diagram:
- Particle Initialization
- Fitness Evaluation
- Velocity & Position Update
- Optimal Hyperparameters Extraction & Model Export
"""

import os
import sys
import argparse

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import find_dataset_root, inspect_dataset, generate_synthetic_openneuro_dataset
from src.segmentation import (
    load_and_preprocess_all_subjects,
    generate_all_windows,
    create_subject_stratified_split
)
from src.pso_optimizer import PSOHyperparameterOptimizer


def main(data_dir=None,
         n_particles=6,
         n_iterations=5,
         eval_epochs=3,
         final_train_epochs=25,
         target_samples=1000,
         test_size=0.2,
         max_subjects=None):
    
    print("=" * 75)
    print("FEATURE 8: PARTICLE SWARM OPTIMIZATION (PSO) HYPERPARAMETER TUNING")
    print("=" * 75)
    
    # 1. Locate Dataset
    if data_dir is None:
        data_dir = find_dataset_root([
            os.path.join(PROJECT_ROOT, "dataset"),
            "/kaggle/input/openneuro-ds004504/ds004504",
            "dataset"
        ])
        
    print(f"Dataset root: {data_dir}")
    inspection = inspect_dataset(data_dir)
    if len(inspection['available_subjects']) == 0:
        print("\n[NOTE] Generating synthetic OpenNeuro ds004504 dataset...")
        generate_synthetic_openneuro_dataset(data_dir, n_subjects=88, duration_sec=120, sfreq=500.0)
        inspection = inspect_dataset(data_dir)
        
    # 2. Preprocess subjects (incorporating Bandpass, Notch, and ICA)
    subject_records = load_and_preprocess_all_subjects(data_dir, target_sfreq=128.0, max_subjects=max_subjects, verbose=False)

    
    # 3. Create anti-leakage train/validation split
    (X_train, y_train, _), (X_val, y_val, _) = create_subject_stratified_split(
        subject_records,
        test_size=test_size,
        random_state=42,
        window_sec=5.0,
        overlap=0.5,
        sfreq=128.0,
        target_total_samples=target_samples
    )
    
    results_dir = os.path.join(PROJECT_ROOT, "results")
    static_img_dir = os.path.join(PROJECT_ROOT, "static", "images", "results")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(static_img_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # 4. Initialize and Run PSO Optimizer
    pso = PSOHyperparameterOptimizer(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_particles=n_particles,
        n_iterations=n_iterations,
        eval_epochs=eval_epochs,
        random_state=42
    )
    
    best_hyperparams, best_fitness, history = pso.optimize(verbose=True)
    
    # 5. Save results and convergence plot
    pso.save_results(output_dir=results_dir, static_dir=static_img_dir)
    
    # 6. Train final model using optimal hyperparameters
    final_ckpt = os.path.join(models_dir, "cnn_lstm_pso_optimized.pt")
    wrapper, train_history = pso.train_final_model(
        epochs=final_train_epochs,
        output_checkpoint_path=final_ckpt
    )
    
    print("\n" + "=" * 75)
    print("PSO OPTIMIZATION AND MODEL TRAINING COMPLETE!")
    print(f"Optimal Hyperparameters: {best_hyperparams}")
    print(f"Model Checkpoint       : {final_ckpt}")
    print(f"Convergence Plot       : {os.path.join(results_dir, 'pso_convergence.png')}")
    print("=" * 75)
    return best_hyperparams, best_fitness


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Particle Swarm Optimization for CNN-LSTM")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to dataset directory")
    parser.add_argument("--n_particles", type=int, default=6, help="Number of particles in swarm")
    parser.add_argument("--n_iterations", type=int, default=5, help="Number of PSO iterations")
    parser.add_argument("--eval_epochs", type=int, default=3, help="Training epochs per particle evaluation")
    parser.add_argument("--final_epochs", type=int, default=25, help="Epochs for training final optimal model")
    parser.add_argument("--samples", type=int, default=600, help="Total EEG window samples")
    parser.add_argument("--max_subjects", type=int, default=None, help="Limit number of subjects processed for fast runs")
    args = parser.parse_args()
    
    main(
        data_dir=args.data_dir,
        n_particles=args.n_particles,
        n_iterations=args.n_iterations,
        eval_epochs=args.eval_epochs,
        final_train_epochs=args.final_epochs,
        target_samples=args.samples,
        max_subjects=args.max_subjects
    )

