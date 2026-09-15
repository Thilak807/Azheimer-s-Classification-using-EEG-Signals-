"""
evaluation.py - Evaluation, Metrics, Confusion Matrices & Plots
Calculates Accuracy, Macro Precision, Recall, and F1-score;
generates and saves Confusion Matrix heatmaps, Training Curves, and comparison CSV.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

CLASS_NAMES = ['AD', 'CN', 'MCI']


def evaluate_experiment(model, X_test, y_test,
                        class_names=CLASS_NAMES,
                        experiment_name="50_50",
                        history=None,
                        results_dir="results"):
    """
    Evaluates the model on untouched test subjects and generates all visual outputs.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Predictions
    y_pred = model.predict(X_test)
    
    # 2. Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    prec_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print("\n" + "=" * 65)
    print(f"EVALUATION RESULTS: EXPERIMENT {experiment_name.upper()}")
    print("=" * 65)
    print(f"Accuracy         : {acc * 100:.2f}%")
    print(f"Precision (Macro): {prec_macro * 100:.2f}% (Weighted: {prec_weighted * 100:.2f}%)")
    print(f"Recall (Macro)   : {rec_macro * 100:.2f}% (Weighted: {rec_weighted * 100:.2f}%)")
    print(f"F1-Score (Macro) : {f1_macro * 100:.2f}% (Weighted: {f1_weighted * 100:.2f}%)")
    print("-" * 65)
    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4, zero_division=0))
    print("=" * 65)
    
    # 3. Confusion Matrix Plot & Save
    cm = confusion_matrix(y_test, y_pred)
    cm_path = os.path.join(results_dir, f"confusion_matrix_{experiment_name}.png")
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix Heatmap ({experiment_name.replace("_", "/")} Split)\nAccuracy: {acc*100:.2f}%', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"  Saved confusion matrix heatmap to: {cm_path}")
    
    # Also sync to static directory for web dashboard
    static_img_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "images", "results")
    if os.path.exists(static_img_dir):
        import shutil
        shutil.copy(cm_path, os.path.join(static_img_dir, os.path.basename(cm_path)))
    
    # 4. Training History Curves Plot & Save
    if history is not None:
        h = history.history if hasattr(history, 'history') else history
        hist_path = os.path.join(results_dir, f"training_history_{experiment_name}.png")
        
        epochs_range = range(1, len(h.get('loss', [])) + 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy curve
        if 'accuracy' in h:
            axes[0].plot(epochs_range, [a * 100 for a in h['accuracy']], 'b-o', label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in h and len(h['val_accuracy']) > 0:
            axes[0].plot(epochs_range, [a * 100 for a in h['val_accuracy']], 'r--s', label='Validation Accuracy', linewidth=2)
        axes[0].set_title(f'Training vs Validation Accuracy ({experiment_name})', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epochs', fontsize=11)
        axes[0].set_ylabel('Accuracy (%)', fontsize=11)
        axes[0].legend(loc='lower right')
        axes[0].grid(True, alpha=0.3)
        
        # Loss curve
        if 'loss' in h:
            axes[1].plot(epochs_range, h['loss'], 'b-o', label='Training Loss', linewidth=2)
        if 'val_loss' in h and len(h['val_loss']) > 0:
            axes[1].plot(epochs_range, h['val_loss'], 'r--s', label='Validation Loss', linewidth=2)
        axes[1].set_title(f'Training vs Validation Loss ({experiment_name})', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epochs', fontsize=11)
        axes[1].set_ylabel('Cross-Entropy Loss', fontsize=11)
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(hist_path, dpi=300)
        plt.close()
        print(f"  Saved training history curves to: {hist_path}")
        if os.path.exists(static_img_dir):
            import shutil
            shutil.copy(hist_path, os.path.join(static_img_dir, os.path.basename(hist_path)))
        
    metrics_dict = {
        'Train/Test': experiment_name.replace('_', '/'),
        'Accuracy': round(acc * 100, 2),
        'Precision': round(prec_macro * 100, 2),
        'Recall': round(rec_macro * 100, 2),
        'F1 Score': round(f1_macro * 100, 2)
    }
    return metrics_dict


def save_summary_comparison_table(results_list, output_csv="results/model_results.csv"):
    """
    Consolidates experiment results into a clean comparison DataFrame and exports to CSV.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df = pd.DataFrame(results_list)
    df.to_csv(output_csv, index=False)
    
    print("\n" + "=" * 65)
    print("FINAL EXPERIMENTAL COMPARISON TABLE")
    print("=" * 65)
    print(df.to_string(index=False))
    print("=" * 65)
    print(f"Saved results summary to: {output_csv}\n")
    return df
