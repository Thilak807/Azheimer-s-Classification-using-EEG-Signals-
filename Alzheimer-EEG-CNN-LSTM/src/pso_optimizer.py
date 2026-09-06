# -*- coding: utf-8 -*-
"""
pso_optimizer.py - Particle Swarm Optimization (PSO) for CNN-LSTM Hyperparameters
Box 8 in Architecture Diagram:
- Particle Initialization (Random Population)
- Fitness Evaluation (Train CNN-LSTM -> Compute Accuracy/Loss)
- Update Velocity & Position (pBest, gBest)
- Stop Condition & Optimal Hyperparameters Export
"""

import os
import json
import time
import copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .model import PyTorchHybridCNNLSTM, UnifiedEEGModelWrapper


# Discrete Parameter Grids
CNN_FILTERS_1_OPTIONS = [32, 64, 96, 128]
CNN_FILTERS_2_OPTIONS = [64, 128, 192, 256]
LSTM_UNITS_OPTIONS = [32, 64, 96, 128]
BATCH_SIZE_OPTIONS = [16, 32, 64]


class Particle:
    """
    Represents a single candidate hyperparameter solution within the swarm.
    Vector dimensions (all normalized in [0, 1]):
      0: log10(lr) [-4.0, -2.0]  -> 1e-4 to 1e-2
      1: dropout_rate [0.15, 0.50]
      2: cnn_filters_1 [32, 64, 96, 128]
      3: cnn_filters_2 [64, 128, 192, 256]
      4: lstm_units [32, 64, 96, 128]
      5: batch_size [16, 32, 64]
    """
    DIM = 6

    def __init__(self, random_state=None):
        rng = np.random.RandomState(random_state)
        # Random initial position in [0, 1]^D
        self.position = rng.uniform(0.05, 0.95, size=self.DIM)
        # Random initial velocity
        self.velocity = rng.uniform(-0.1, 0.1, size=self.DIM)
        
        self.pbest_position = np.copy(self.position)
        self.pbest_fitness = -float('inf')
        self.current_fitness = -float('inf')

    def to_hyperparameters(self):
        """Converts continuous [0, 1] position coordinates to model hyperparameters."""
        # 0: Learning Rate (log-scale 10^-4 to 10^-2)
        log_lr = -4.0 + self.position[0] * 2.0
        lr = float(10.0 ** log_lr)

        # 1: Dropout rate (0.15 to 0.50)
        dropout = float(0.15 + self.position[1] * 0.35)

        # 2: CNN Filters 1 (index into options)
        idx_f1 = min(len(CNN_FILTERS_1_OPTIONS) - 1, int(self.position[2] * len(CNN_FILTERS_1_OPTIONS)))
        filters_1 = CNN_FILTERS_1_OPTIONS[idx_f1]

        # 3: CNN Filters 2
        idx_f2 = min(len(CNN_FILTERS_2_OPTIONS) - 1, int(self.position[3] * len(CNN_FILTERS_2_OPTIONS)))
        filters_2 = CNN_FILTERS_2_OPTIONS[idx_f2]

        # 4: LSTM Units
        idx_lstm = min(len(LSTM_UNITS_OPTIONS) - 1, int(self.position[4] * len(LSTM_UNITS_OPTIONS)))
        lstm_units = LSTM_UNITS_OPTIONS[idx_lstm]

        # 5: Batch size
        idx_bs = min(len(BATCH_SIZE_OPTIONS) - 1, int(self.position[5] * len(BATCH_SIZE_OPTIONS)))
        batch_size = BATCH_SIZE_OPTIONS[idx_bs]

        return {
            'lr': round(lr, 6),
            'dropout_rate': round(dropout, 3),
            'cnn_filters_1': filters_1,
            'cnn_filters_2': filters_2,
            'lstm_units': lstm_units,
            'batch_size': batch_size
        }


class PSOHyperparameterOptimizer:
    """
    Particle Swarm Optimization Engine for Hybrid CNN-LSTM Architecture.
    Implements:
    - Canonical PSO velocity & position updates
    - Swarm inertia weight linear decay
    - Acceleration coefficients (cognitive c1, social c2)
    - Early stopping & convergence tracking
    - Visualization plot generation
    """
    def __init__(self, X_train, y_train, X_val, y_val,
                 n_particles=6,
                 n_iterations=5,
                 eval_epochs=3,
                 w_start=0.9,
                 w_end=0.4,
                 c1=1.494,
                 c2=1.494,
                 v_max=0.2,
                 device=None,
                 random_state=42):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.eval_epochs = eval_epochs
        
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.rng = np.random.RandomState(random_state)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.particles = [Particle(random_state=random_state + i) for i in range(n_particles)]
        self.gbest_position = None
        self.gbest_fitness = -float('inf')
        self.gbest_hyperparams = None
        
        self.history = {
            'iteration': [],
            'gbest_fitness': [],
            'avg_fitness': [],
            'particles': []
        }

    def evaluate_particle_fitness(self, particle):
        """
        Evaluates a particle's hyperparameter set by training a candidate CNN-LSTM
        model for a short budget (eval_epochs) and measuring validation accuracy & loss.
        Fitness = Val_Acc * 0.85 + (1.0 - min(1.0, Val_Loss)) * 0.15
        """
        hp = particle.to_hyperparameters()
        
        try:
            n_channels = self.X_train.shape[2]
            win_len = self.X_train.shape[1]
            num_classes = len(np.unique(np.concatenate([self.y_train, self.y_val])))
            
            model = PyTorchHybridCNNLSTM(
                n_channels=n_channels,
                window_length=win_len,
                num_classes=num_classes,
                filters_1=hp['cnn_filters_1'],
                filters_2=hp['cnn_filters_2'],
                lstm_units=hp['lstm_units'],
                dropout_rate=hp['dropout_rate']
            ).to(self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'], weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()
            
            train_ds = TensorDataset(
                torch.tensor(self.X_train, dtype=torch.float32),
                torch.tensor(self.y_train, dtype=torch.long)
            )
            loader = DataLoader(train_ds, batch_size=hp['batch_size'], shuffle=True)
            
            vx = torch.tensor(self.X_val, dtype=torch.float32).to(self.device)
            vy = torch.tensor(self.y_val, dtype=torch.long).to(self.device)
            
            # Fast training loop
            for epoch in range(self.eval_epochs):
                model.train()
                for bx, by in loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    optimizer.zero_grad()
                    out = model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()
                    
            # Evaluate on validation split
            model.eval()
            with torch.no_grad():
                vout = model(vx)
                v_loss = criterion(vout, vy).item()
                preds = torch.argmax(vout, dim=1)
                v_acc = (preds == vy).float().mean().item()
                
            # Composite fitness score: maximize accuracy while minimizing loss
            fitness = (v_acc * 0.85) + (max(0.0, 1.0 - min(1.0, v_loss)) * 0.15)
            return fitness, v_acc, v_loss
            
        except Exception as e:
            print(f"    [PSO Warning] Particle evaluation failed with error: {e}")
            return 0.0, 0.0, 5.0

    def optimize(self, verbose=True):
        """
        Executes the main Particle Swarm Optimization loop:
        1. Evaluates all particles.
        2. Updates personal best (pBest) and global best (gBest).
        3. Updates velocities and positions.
        4. Clamps positions within [0, 1].
        """
        if verbose:
            print("\n" + "=" * 70)
            print("PARTICLE SWARM OPTIMIZATION (PSO) - HYPERPARAMETER TUNING")
            print("=" * 70)
            print(f"Swarm Size: {self.n_particles} | Iterations: {self.n_iterations} | Eval Epochs/Particle: {self.eval_epochs}")
            print("Search Space: LR [1e-4..1e-2], Dropout [0.15..0.50], Filters [32..256], LSTM [32..128]")
            print("-" * 70)
            
        start_time = time.time()
        
        for it in range(self.n_iterations):
            it_start = time.time()
            # Linearly decaying inertia weight
            w = self.w_start - ((self.w_start - self.w_end) * (it / max(1, self.n_iterations - 1)))
            
            fitness_scores = []
            it_particle_records = []
            
            for p_idx, particle in enumerate(self.particles):
                fitness, v_acc, v_loss = self.evaluate_particle_fitness(particle)
                particle.current_fitness = fitness
                fitness_scores.append(fitness)
                
                # Update Personal Best (pBest)
                if fitness > particle.pbest_fitness:
                    particle.pbest_fitness = fitness
                    particle.pbest_position = np.copy(particle.position)
                    
                # Update Global Best (gBest)
                if fitness > self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest_position = np.copy(particle.position)
                    self.gbest_hyperparams = particle.to_hyperparameters()
                    if verbose:
                        print(f"  ? [New gBest Found! Iter {it+1}, Particle {p_idx+1}] "
                              f"Fitness: {fitness:.4f} (Acc: {v_acc*100:.1f}%, Loss: {v_loss:.3f}) | {self.gbest_hyperparams}")
                              
                it_particle_records.append({
                    'particle_idx': p_idx,
                    'fitness': float(fitness),
                    'val_acc': float(v_acc),
                    'val_loss': float(v_loss),
                    'params': particle.to_hyperparameters()
                })
                
            avg_fit = float(np.mean(fitness_scores))
            self.history['iteration'].append(it + 1)
            self.history['gbest_fitness'].append(float(self.gbest_fitness))
            self.history['avg_fitness'].append(avg_fit)
            self.history['particles'].append(it_particle_records)
            
            if verbose:
                it_dur = time.time() - it_start
                print(f"Iteration {it+1:02d}/{self.n_iterations:02d} completed in {it_dur:.1f}s | "
                      f"Avg Fitness: {avg_fit:.4f} | gBest Fitness: {self.gbest_fitness:.4f}")
                      
            # Update Velocities & Positions for next iteration
            for particle in self.particles:
                r1 = self.rng.uniform(0.0, 1.0, size=Particle.DIM)
                r2 = self.rng.uniform(0.0, 1.0, size=Particle.DIM)
                
                # Canonical PSO Velocity equation:
                # v = w*v + c1*r1*(pBest - x) + c2*r2*(gBest - x)
                cognitive = self.c1 * r1 * (particle.pbest_position - particle.position)
                social = self.c2 * r2 * (self.gbest_position - particle.position)
                new_v = w * particle.velocity + cognitive + social
                
                # Clamp velocity within [-v_max, v_max]
                new_v = np.clip(new_v, -self.v_max, self.v_max)
                particle.velocity = new_v
                
                # Position update: x = x + v
                particle.position = np.clip(particle.position + new_v, 0.0, 1.0)
                
        total_time = time.time() - start_time
        
        if verbose:
            print("=" * 70)
            print("PSO OPTIMIZATION COMPLETED SUCCESSFULLY!")
            print(f"Total Time       : {total_time:.1f}s")
            print(f"Best Fitness     : {self.gbest_fitness:.4f}")
            print(f"Optimal Params   : {self.gbest_hyperparams}")
            print("=" * 70 + "\n")
            
        return self.gbest_hyperparams, self.gbest_fitness, self.history

    def plot_convergence(self, output_path):
        """Generates a publication-grade PSO convergence and fitness trajectory curve."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        iterations = self.history['iteration']
        gbest = self.history['gbest_fitness']
        avg = self.history['avg_fitness']
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(iterations, gbest, 'o-', color='#2ecc71', linewidth=2.5, markersize=7, label='Global Best (gBest) Fitness')
        ax.plot(iterations, avg, 's--', color='#3498db', linewidth=2.0, markersize=6, label='Swarm Average Fitness')
        
        ax.set_title('Particle Swarm Optimization (PSO) Convergence Curve\nHybrid CNN-LSTM Hyperparameter Search', 
                     fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel('PSO Iteration', fontsize=11, fontweight='bold')
        ax.set_ylabel('Fitness Score', fontsize=11, fontweight='bold')
        ax.set_ylim(min(min(avg), 0.5) - 0.05, 1.05)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='lower right', frameon=True, shadow=True)
        
        # Annotate best fitness value
        if len(gbest) > 0:
            final_best = gbest[-1]
            ax.annotate(f'Optimal: {final_best:.4f}',
                        xy=(iterations[-1], final_best),
                        xytext=(iterations[-1] - 0.8, final_best - 0.08),
                        arrowprops=dict(facecolor='#27ae60', shrink=0.08, width=1.5, headwidth=6),
                        fontsize=10, fontweight='bold', color='#27ae60')
                        
        plt.tight_layout()
        plt.savefig(output_path, dpi=250)
        plt.close()
        print(f"  [PSO] Saved convergence plot to: {output_path}")

    def save_results(self, output_dir, static_dir=None):
        """Saves optimal parameters JSON, history, and convergence plot."""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save JSON
        json_path = os.path.join(output_dir, "pso_optimal_hyperparameters.json")
        out_data = {
            'optimal_hyperparameters': self.gbest_hyperparams,
            'best_fitness': float(self.gbest_fitness),
            'history': {
                'iteration': self.history['iteration'],
                'gbest_fitness': self.history['gbest_fitness'],
                'avg_fitness': self.history['avg_fitness']
            }
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(out_data, f, indent=4)
        print(f"  [PSO] Saved optimal hyperparameters JSON: {json_path}")
        
        # 2. Save Plot
        plot_path = os.path.join(output_dir, "pso_convergence.png")
        self.plot_convergence(plot_path)
        
        # 3. Sync to static dir if available
        if static_dir and os.path.exists(static_dir):
            import shutil
            shutil.copy(plot_path, os.path.join(static_dir, "pso_convergence.png"))
            
        return json_path, plot_path

    def train_final_model(self, epochs=25, output_checkpoint_path="models/cnn_lstm_pso_optimized.pt"):
        """
        Trains a full Hybrid CNN-LSTM model with the optimal hyperparameters found by PSO
        and saves the checkpoint for instant inference in the Flask app.
        """
        if self.gbest_hyperparams is None:
            raise RuntimeError("PSO must be run (.optimize()) before training final model.")
            
        hp = self.gbest_hyperparams
        print("\n" + "=" * 65)
        print("TRAINING FINAL MODEL WITH PSO-OPTIMIZED HYPERPARAMETERS")
        print("=" * 65)
        print(f"Optimal Hyperparameters: {hp}")
        print(f"Epochs: {epochs} | Batch Size: {hp['batch_size']}")
        
        input_shape = (self.X_train.shape[1], self.X_train.shape[2])
        num_classes = len(np.unique(np.concatenate([self.y_train, self.y_val])))
        
        wrapper = UnifiedEEGModelWrapper(
            input_shape=input_shape,
            num_classes=num_classes,
            lr=hp['lr'],
            framework='torch',
            hyperparams=hp
        )
        
        history = wrapper.fit(
            self.X_train, self.y_train,
            X_val=self.X_val, y_val=self.y_val,
            epochs=epochs,
            batch_size=hp['batch_size']
        )
        
        os.makedirs(os.path.dirname(os.path.abspath(output_checkpoint_path)), exist_ok=True)
        wrapper.save(output_checkpoint_path)
        print(f"Final PSO-optimized model saved to: {output_checkpoint_path}\n")
        return wrapper, history
