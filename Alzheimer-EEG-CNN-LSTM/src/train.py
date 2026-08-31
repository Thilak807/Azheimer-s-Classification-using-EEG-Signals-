"""
train.py - Model Training Engine
Handles model instantiation, class weight computation for imbalance mitigation,
training loop execution, and checkpoint saving.
"""

import os
import time
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from .model import UnifiedEEGModelWrapper, HAS_KERAS

if HAS_KERAS:
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def compute_balanced_class_weights(y_train, num_classes=3):
    """
    Computes class weights to address dataset imbalance between AD, HC, and FTD.
    """
    classes = np.arange(num_classes)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    weight_dict = {i: float(w) for i, w in enumerate(weights)}
    print(f"Computed Class Weights: {weight_dict}")
    return weight_dict


def train_experiment(X_train, y_train, X_val, y_val,
                     experiment_name="50_50",
                     epochs=40,
                     batch_size=32,
                     lr=0.001,
                     models_dir="models",
                     framework='auto'):
    """
    Executes training for a specific train/test split experiment.
    """
    input_shape = (X_train.shape[1], X_train.shape[2]) # (window_length, n_channels)
    num_classes = len(np.unique(np.concatenate([y_train, y_val])))
    
    print("\n" + "=" * 65)
    print(f"STARTING TRAINING: EXPERIMENT {experiment_name.upper()}")
    print(f"Input Shape: {input_shape} | Classes: {num_classes}")
    print(f"Training Samples: {len(y_train)} | Validation Samples: {len(y_val)}")
    print("=" * 65)
    
    # Class weights
    class_weights = compute_balanced_class_weights(y_train, num_classes=num_classes)
    
    # Instantiate Unified Model
    model = UnifiedEEGModelWrapper(input_shape, num_classes=num_classes, lr=lr, framework=framework)
    
    callbacks_list = []
    ckpt_path = os.path.join(models_dir, f"cnn_lstm_{experiment_name}.keras")
    
    if model.framework == 'keras' and HAS_KERAS:
        callbacks_list = [
            EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1),
            ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, verbose=0)
        ]
        
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks_list=callbacks_list
    )
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f}s.")
    
    # Save the trained model
    save_path = os.path.join(models_dir, f"cnn_lstm_{experiment_name}")
    model.save(save_path)
    
    return model, history, elapsed
