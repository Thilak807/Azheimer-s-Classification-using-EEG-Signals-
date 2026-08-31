"""
model.py - Hybrid CNN-LSTM Neural Network Architecture
Implements 1D CNN for spatial/local temporal EEG feature extraction coupled with
LSTM for sequence modeling and a softmax classification head.
Supports both TensorFlow/Keras and PyTorch backends seamlessly.
"""

import os
import numpy as np

# Detect available deep learning frameworks
HAS_KERAS = False
HAS_TORCH = False

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, callbacks
    HAS_KERAS = True
except ImportError:
    pass

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    pass


def build_keras_cnn_lstm_model(input_shape, num_classes=3, lr=0.001):
    """
    Builds the Keras/TensorFlow Hybrid CNN-LSTM Model.
    Architecture:
      Input (window_length, n_channels)
         ?
      Conv1D(64, k=7) -> BatchNorm -> MaxPool(2) -> Dropout(0.3)
         ?
      Conv1D(128, k=5) -> BatchNorm -> MaxPool(2) -> Dropout(0.3)
         ?
      Conv1D(128, k=3) -> BatchNorm -> MaxPool(2) -> Dropout(0.3)
         ?
      Bidirectional LSTM(64) -> Dropout(0.4)
         ?
      Dense(64, relu) -> BatchNorm -> Dropout(0.3)
         ?
      Dense(num_classes, softmax)
    """
    if not HAS_KERAS:
        raise ImportError("TensorFlow/Keras is not installed in the environment.")
        
    model = models.Sequential([
        # Input Layer
        layers.Input(shape=input_shape),
        
        # Block 1: 1D CNN
        layers.Conv1D(filters=64, kernel_size=7, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Block 2: 1D CNN
        layers.Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Block 3: 1D CNN
        layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Block 4: LSTM Layer
        layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
        layers.Dropout(0.4),
        
        # Block 5: Dense Representation
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Output Classification Layer
        layers.Dense(num_classes, activation='softmax')
    ], name="Hybrid_CNN_LSTM_Alzheimer_Classifier")
    
    optimizer = optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


if HAS_TORCH:
    class PyTorchHybridCNNLSTM(nn.Module):
        """
        PyTorch Implementation of the Hybrid CNN-LSTM Model.
        """
        def __init__(self, n_channels=19, window_length=640, num_classes=3):
            super(PyTorchHybridCNNLSTM, self).__init__()
            
            # 1D CNN Feature Extractor (Input: [batch, n_channels, window_length])
            self.conv1 = nn.Sequential(
                nn.Conv1d(in_channels=n_channels, out_channels=64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(0.3)
            )
            self.conv2 = nn.Sequential(
                nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(0.3)
            )
            self.conv3 = nn.Sequential(
                nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(0.3)
            )
            
            # Bidirectional LSTM Layer
            self.lstm = nn.LSTM(
                input_size=128,
                hidden_size=64,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            self.lstm_dropout = nn.Dropout(0.4)
            
            # Fully Connected Classifier
            self.fc1 = nn.Sequential(
                nn.Linear(64 * 2, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            self.fc_out = nn.Linear(64, num_classes)
            
        def forward(self, x):
            # Input shape: [batch, window_length, n_channels] -> transpose to [batch, n_channels, window_length]
            x = x.transpose(1, 2)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            
            # Transpose for LSTM: [batch, 128, reduced_time] -> [batch, reduced_time, 128]
            x = x.transpose(1, 2)
            lstm_out, _ = self.lstm(x)
            
            # Take last time step or mean pool across time
            last_step = lstm_out[:, -1, :]
            last_step = self.lstm_dropout(last_step)
            
            dense = self.fc1(last_step)
            logits = self.fc_out(dense)
            return logits


class UnifiedEEGModelWrapper:
    """
    Unified model interface providing standard .fit(), .predict(), .evaluate(), and .save() methods
    regardless of whether Keras or PyTorch is running underneath.
    """
    def __init__(self, input_shape, num_classes=3, lr=0.001, framework='auto'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lr = lr
        
        if framework == 'auto':
            if HAS_KERAS:
                self.framework = 'keras'
            elif HAS_TORCH:
                self.framework = 'torch'
            else:
                raise RuntimeError("Neither TensorFlow nor PyTorch is available.")
        else:
            self.framework = framework
            
        if self.framework == 'keras':
            self.model = build_keras_cnn_lstm_model(input_shape, num_classes=num_classes, lr=lr)
        else:
            n_channels = input_shape[1]
            win_len = input_shape[0]
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = PyTorchHybridCNNLSTM(n_channels=n_channels, window_length=win_len, num_classes=num_classes).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
            self.criterion = nn.CrossEntropyLoss()

    def summary(self):
        print("\n" + "=" * 65)
        print(f"HYBRID CNN-LSTM MODEL ARCHITECTURE ({self.framework.upper()} BACKEND)")
        print("=" * 65)
        if self.framework == 'keras':
            self.model.summary()
        else:
            print(self.model)
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"\nTotal Parameters    : {total_params:,}")
            print(f"Trainable Parameters: {trainable_params:,}")
        print("=" * 65 + "\n")

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32, class_weight=None, callbacks_list=None):
        if self.framework == 'keras':
            return self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val) if X_val is not None else None,
                epochs=epochs,
                batch_size=batch_size,
                class_weight=class_weight,
                callbacks=callbacks_list,
                verbose=1
            )
        else:
            # PyTorch training loop
            history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
            
            # Setup weights if provided
            if class_weight is not None:
                weights = [class_weight[i] for i in range(self.num_classes)]
                weight_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
                self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
                
            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            best_val_loss = float('inf')
            patience = 12
            patience_counter = 0
            best_state = None
            
            for epoch in range(epochs):
                self.model.train()
                total_loss, correct, total = 0.0, 0, 0
                for bx, by in train_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    self.optimizer.zero_grad()
                    out = self.model(bx)
                    loss = self.criterion(out, by)
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item() * bx.size(0)
                    preds = torch.argmax(out, dim=1)
                    correct += (preds == by).sum().item()
                    total += bx.size(0)
                    
                train_loss = total_loss / total
                train_acc = correct / total
                history['loss'].append(train_loss)
                history['accuracy'].append(train_acc)
                
                # Validation
                if X_val is not None and y_val is not None:
                    self.model.eval()
                    v_loss, v_corr, v_tot = 0.0, 0, 0
                    with torch.no_grad():
                        vx = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                        vy = torch.tensor(y_val, dtype=torch.long).to(self.device)
                        vout = self.model(vx)
                        v_loss = self.criterion(vout, vy).item()
                        v_preds = torch.argmax(vout, dim=1)
                        v_corr = (v_preds == vy).sum().item()
                        v_tot = vy.size(0)
                        
                    val_loss = v_loss
                    val_acc = v_corr / v_tot
                    history['val_loss'].append(val_loss)
                    history['val_accuracy'].append(val_acc)
                    
                    if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
                        print(f"Epoch {epoch+1:02d}/{epochs:02d} - Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - Val_Loss: {val_loss:.4f} - Val_Acc: {val_acc:.4f}")
                        
                    # Early stopping tracking
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_state = self.model.state_dict().copy()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"Early stopping triggered at epoch {epoch+1}.")
                            break
                else:
                    if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
                        print(f"Epoch {epoch+1:02d}/{epochs:02d} - Loss: {train_loss:.4f} - Acc: {train_acc:.4f}")
                        
            if best_state is not None:
                self.model.load_state_dict(best_state)
                
            # Create a simple History mock object
            class HistoryMock:
                def __init__(self, h):
                    self.history = h
            return HistoryMock(history)

    def predict_proba(self, X):
        if self.framework == 'keras':
            return self.model.predict(X, verbose=0)
        else:
            self.model.eval()
            with torch.no_grad():
                tx = torch.tensor(X, dtype=torch.float32).to(self.device)
                logits = self.model(tx)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def save(self, filepath):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        if self.framework == 'keras':
            # Save as .keras or .h5
            if not filepath.endswith('.keras') and not filepath.endswith('.h5'):
                filepath = filepath + '.keras'
            self.model.save(filepath)
        else:
            if not filepath.endswith('.pt') and not filepath.endswith('.pth'):
                filepath = filepath + '.pt'
            torch.save(self.model.state_dict(), filepath)
        print(f"  Saved trained model to: {filepath}")
