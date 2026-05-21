import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP
from torch.nn import GRUCell, LayerNorm
import os
import time
import seaborn as sns
from scipy import stats
import sys

# ---------------------------
# Data Preparation
# ---------------------------
class RCBeamDataset:
    def __init__(self, data_path):
        # Read data with specified column names
        self.raw_data = pd.read_csv(data_path, header=None, names=[
            'b', 'h', "bp", "hp", 'fc', 'ALfyL', 'Atfyt/s', 'Ts'
        ]).astype(np.float32)
        self._preprocess()

    def _preprocess(self):
        # Derived features aligned with physics anchor
        self.raw_data['A0'] = (0.85 * self.raw_data['bp'] * self.raw_data['hp']).astype(np.float32)
        self.raw_data['ph'] = (2 * (self.raw_data['bp'] + self.raw_data['hp'])).astype(np.float32)
        self.raw_data['sqrt_fc'] = np.sqrt(self.raw_data['fc']).astype(np.float32)

        # Node-wise feature groups
        self.feature_groups = {
            'stirrup': ["Atfyt/s", "bp", "hp", "A0", "ph"],
            'longitudinal': ["ALfyL"],
            'concrete': ["fc", "b", "h", "sqrt_fc"]
        }

        # Train/val/test split
        train_val, test = train_test_split(self.raw_data, test_size=0.2, random_state=42)
        train, val = train_test_split(train_val, test_size=0.125, random_state=42)

        # Standard scalers
        self.scalers = {
            'stirrup': StandardScaler().fit(train[self.feature_groups['stirrup']]),
            'longitudinal': StandardScaler().fit(train[self.feature_groups['longitudinal']]),
            'concrete': StandardScaler().fit(train[self.feature_groups['concrete']]),
            'target': StandardScaler().fit(train[['Ts']])
        }

        # Normalized datasets
        self.datasets = {
            'train': self._normalize_data(train),
            'val': self._normalize_data(val),
            'test': self._normalize_data(test)
        }

    def _normalize_data(self, df):
        return {
            'stirrup': self.scalers['stirrup'].transform(df[self.feature_groups['stirrup']]).astype(np.float32),
            'longitudinal': self.scalers['longitudinal'].transform(df[self.feature_groups['longitudinal']]).astype(np.float32),
            'concrete': self.scalers['concrete'].transform(df[self.feature_groups['concrete']]).astype(np.float32),
            'target': self.scalers['target'].transform(df[['Ts']]).astype(np.float32),
            'original': df.copy().astype(np.float32)
        }

    def get_loaders(self, batch_size=32):
        return {
            phase: DataLoader(
                [self._create_hetero_data(idx, phase) for idx in range(len(self.datasets[phase]['target']))],
                batch_size=batch_size,
                shuffle=(phase == 'train'),
                follow_batch=['stirrup', 'longitudinal', 'concrete']
            )
            for phase in ['train', 'val', 'test']
        }

    def _create_hetero_data(self, idx, phase):
        data = HeteroData()

        # One node per type (per specimen graph)
        for node_type in ['stirrup', 'longitudinal', 'concrete']:
            features = self.datasets[phase][node_type][idx].reshape(1, -1)
            data[node_type].x = torch.tensor(features, dtype=torch.float32)

            # Unnormalized originals for physics anchor
            orig_features = self.datasets[phase]['original'].iloc[idx][self.feature_groups[node_type]].values.reshape(1, -1)
            data[node_type].orig = torch.tensor(orig_features, dtype=torch.float32)

            # Self-edge for numerical stability (not used by typed messages)
            data[node_type].edge_index = torch.tensor([[0], [0]], dtype=torch.long)

        # Typed edges: s→l, l→c, s→c
        data['stirrup', 'confines', 'longitudinal'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['longitudinal', 'transfers', 'concrete'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['stirrup', 'confines', 'concrete'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)

        # Tracking
        data.idx = idx
        data.phase = phase

        # Target (normalized)
        data.y = torch.tensor(self.datasets[phase]['target'][idx], dtype=torch.float32)
        return data


# ---------------------------
# GNN: Typed messages + typed updates
# ---------------------------
class EnhancedGNNLayer(nn.Module):
    """
    Messages (edge-specific):
      s→l : MLP_s2l([h_s, h_l])
      l→c : MLP_l2c([h_l, h_c])
      s→c : MLP_s2c([h_s, h_c])
    Updates:
      h_s^{k+1} = GRU_s(a_s^{k}, h_s^{k})       # a_s is zero here (no incoming to s)
      h_l^{k+1} = GRU_l(a_l^{k}, h_l^{k})
      h_c^{k+1} = MLP_upd([h_c^{k}, a_c^{k}])
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # edge-type message MLPs
        self.msg_s2l = MLP([2*hidden_dim, 2*hidden_dim, hidden_dim], norm="batch_norm")
        self.msg_l2c = MLP([2*hidden_dim, 2*hidden_dim, hidden_dim], norm="batch_norm")
        self.msg_s2c = MLP([2*hidden_dim, 2*hidden_dim, hidden_dim], norm="batch_norm")

        # node-type update units
        self.gru_s = GRUCell(hidden_dim, hidden_dim)
        self.gru_l = GRUCell(hidden_dim, hidden_dim)
        self.mlp_upd_c = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # norms
        self.norm_s = LayerNorm(hidden_dim)
        self.norm_l = LayerNorm(hidden_dim)
        self.norm_c = LayerNorm(hidden_dim)

    def forward(self, x_dict, edge_index_dict):
        h_s = x_dict['stirrup']
        h_l = x_dict['longitudinal']
        h_c = x_dict['concrete']

        # aggregated messages (sum) to each destination
        a_s = torch.zeros_like(h_s)  # no incoming to 'stirrup' in this graph
        a_l = torch.zeros_like(h_l)
        a_c = torch.zeros_like(h_c)

        # s→l
        key = ('stirrup', 'confines', 'longitudinal')
        if key in edge_index_dict:
            ei = edge_index_dict[key]
            m = self.msg_s2l(torch.cat([h_s[ei[0]], h_l[ei[1]]], dim=-1))
            a_l = a_l.index_add(0, ei[1], m)

        # l→c
        key = ('longitudinal', 'transfers', 'concrete')
        if key in edge_index_dict:
            ei = edge_index_dict[key]
            m = self.msg_l2c(torch.cat([h_l[ei[0]], h_c[ei[1]]], dim=-1))
            a_c = a_c.index_add(0, ei[1], m)

        # s→c
        key = ('stirrup', 'confines', 'concrete')
        if key in edge_index_dict:
            ei = edge_index_dict[key]
            m = self.msg_s2c(torch.cat([h_s[ei[0]], h_c[ei[1]]], dim=-1))
            a_c = a_c.index_add(0, ei[1], m)

        # node updates + residual + norm
        h_s_new = self.gru_s(a_s, h_s)
        h_s_out = self.norm_s(h_s + h_s_new)

        h_l_new = self.gru_l(a_l, h_l)
        h_l_out = self.norm_l(h_l + h_l_new)

        h_c_new = self.mlp_upd_c(torch.cat([h_c, a_c], dim=-1))
        h_c_out = self.norm_c(h_c + h_c_new)

        return {'stirrup': h_s_out, 'longitudinal': h_l_out, 'concrete': h_c_out}


class PhysicsGNN(nn.Module):
    """
    Encoders → K message-passing layers → physics-guided readout:
      T_pred = α·t_c(h_c^K) + β·t_s(h_s^K) + γ·t_l(h_l^K) + ε·t_int(h_s^K,h_l^K,h_c^K)
    with α,β,γ,ε ≥ 0 (enforced via softplus).
    """
    def __init__(self, hidden_dim=512, num_layers=6):
        super().__init__()
        # feature encoders
        self.encoders = nn.ModuleDict({
            'stirrup': nn.Sequential(
                MLP([5, hidden_dim, hidden_dim], norm="batch_norm"),
                nn.ReLU(), nn.Dropout(0.3)
            ),
            'longitudinal': nn.Sequential(
                MLP([1, hidden_dim, hidden_dim], norm="batch_norm"),
                nn.ReLU(), nn.Dropout(0.3)
            ),
            'concrete': nn.Sequential(
                MLP([4, hidden_dim, hidden_dim], norm="batch_norm"),
                nn.ReLU(), nn.Dropout(0.3)
            )
        })

        # GNN layers
        self.layers = nn.ModuleList([EnhancedGNNLayer(hidden_dim) for _ in range(num_layers)])

        # code-aligned readout heads
        self.t_s = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 1))
        self.t_l = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 1))
        self.t_c = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 1))
        self.t_int = nn.Sequential(  # small interaction
            nn.Linear(3*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # nonnegative weights via softplus
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_beta  = nn.Parameter(torch.tensor(0.0))
        self.log_gamma = nn.Parameter(torch.tensor(0.0))
        self.log_eps   = nn.Parameter(torch.tensor(-4.0))

    def nonneg_weights(self):
        return (F.softplus(self.log_alpha),
                F.softplus(self.log_beta),
                F.softplus(self.log_gamma),
                F.softplus(self.log_eps))

    def epsilon(self):
        return self.nonneg_weights()[-1]

    def forward(self, data):
        # encode
        h = {
            'stirrup':      self.encoders['stirrup'](data['stirrup'].x),
            'longitudinal': self.encoders['longitudinal'](data['longitudinal'].x),
            'concrete':     self.encoders['concrete'](data['concrete'].x),
        }
        # message passing
        for layer in self.layers:
            h = layer(h, data.edge_index_dict)

        # readout
        t_s = self.t_s(h['stirrup']).squeeze(-1)
        t_l = self.t_l(h['longitudinal']).squeeze(-1)
        t_c = self.t_c(h['concrete']).squeeze(-1)
        t_int = self.t_int(torch.cat([h['stirrup'], h['longitudinal'], h['concrete']], dim=-1)).squeeze(-1)

        alpha, beta, gamma, eps = self.nonneg_weights()
        T_pred = alpha*t_c + beta*t_s + gamma*t_l + eps*t_int
        return T_pred  # still in normalized target space


# ---------------------------
# Training Framework
# ---------------------------
class TorsionTrainer:
    def __init__(self, data_path):
        self.dataset = RCBeamDataset(data_path)
        self.loaders = self.dataset.get_loaders()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.model = PhysicsGNN(hidden_dim=512, num_layers=6).to(self.device)

        # ---- GRU diagnostics: tracking containers + forward hooks (no arch changes) ----
        self._epoch_hidden_norm_s = []
        self._epoch_hidden_norm_l = []
        self._epoch_avg_pred_torsion = []
        self._train_batch_hidden_s = []
        self._train_batch_hidden_l = []

        def _hook_s(module, inputs, output):
            self._train_batch_hidden_s.append(output.detach().cpu())

        def _hook_l(module, inputs, output):
            self._train_batch_hidden_l.append(output.detach().cpu())

        last_layer = self.model.layers[-1]
        self._gru_hook_handles = [
            last_layer.gru_s.register_forward_hook(_hook_s),
            last_layer.gru_l.register_forward_hook(_hook_l),
        ]
        # -------------------------------------------------------------------------------

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=15, verbose=True
        )
        self.loss_history = {'train': [], 'val': [], 'test': []}
        self.train_rmse_history, self.val_rmse_history, self.test_rmse_history = [], [], []
        self.best_val_loss = float('inf')
        self.best_model_state = None

        # Normalization params for un/renorm
        self.scale = torch.tensor(
            self.dataset.scalers['target'].scale_.astype(np.float32),
            dtype=torch.float32, device=self.device
        )
        self.mean = torch.tensor(
            self.dataset.scalers['target'].mean_.astype(np.float32),
            dtype=torch.float32, device=self.device
        )

        # Physics/interaction regularization
        self.lambda_int = 1e-4  # small penalty on ε to prefer additivity unless data needs interaction

        # Data storage for plots/CSVs
        self.training_history_data = {'epoch': [], 'train_loss': [], 'val_loss': [], 'test_loss': []}
        self.rmse_propagation_data = {'epoch': [], 'train_rmse': [], 'val_rmse': [], 'test_rmse': []}
        self.residual_analysis_data = {}

    def _physics_loss(self, pred, batch):
        # original (unnormalized) features
        fc = batch['concrete'].orig[:, 0].float()  # fc (MPa)
        bp = batch['stirrup'].orig[:, 1].float()   # b'
        hp = batch['stirrup'].orig[:, 2].float()   # h'
        Atfyt_s = batch['stirrup'].orig[:, 0].float()  # At fyt / s
        ALfyL = batch['longitudinal'].orig[:, 0].float()  # AL fyl

        # Derived
        A0 = 0.85 * bp * hp
        ph = 2 * (bp + hp)

        # Mechanics-based anchor (kN·m after /1e6)
        T_concrete = (0.33 * torch.sqrt(fc) * (A0 ** 2) / ph) / 1e6
        T_stirrup  = (2 * A0 * Atfyt_s) / 1e6
        T_long     = (2 * A0 * ALfyL / ph) / 1e6

        T_phys = T_concrete + T_stirrup + T_long
        T_phys_normalized = (T_phys - self.mean) / self.scale

        return F.huber_loss(pred, T_phys_normalized, delta=0.5)

    def train(self, epochs=300):
        start_time = time.time()
        early_stop_counter = 0
        patience = 30

        for epoch in range(epochs):
            physics_weight = max(0.1, 1.5 - epoch / 100)

            # ---- Training ----
            self.model.train()
            train_loss = 0.0
            train_preds, train_trues = [], []

            for batch in self.loaders['train']:
                self.optimizer.zero_grad()
                batch = batch.to(self.device)
                pred = self.model(batch)

                # losses
                loss_data = F.huber_loss(pred, batch.y, delta=0.5)
                loss_physics = self._physics_loss(pred, batch)
                loss_eps = (self.model.epsilon() ** 2) * self.lambda_int

                loss = loss_data + physics_weight * loss_physics + loss_eps
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item()

                # unnormalize for RMSE
                with torch.no_grad():
                    pred_unnorm = pred * self.scale + self.mean
                    true_unnorm = batch.y * self.scale + self.mean
                    train_preds.append(pred_unnorm.cpu().numpy())
                    train_trues.append(true_unnorm.cpu().numpy())

            avg_train_loss = train_loss / len(self.loaders['train'])
            self.loss_history['train'].append(avg_train_loss)

            train_preds = np.concatenate(train_preds)
            train_trues = np.concatenate(train_trues)
            train_rmse = np.sqrt(np.mean((train_preds - train_trues) ** 2))
            self.train_rmse_history.append(train_rmse)

            # ---- Validation ----
            self.model.eval()
            val_loss = 0.0
            val_preds, val_trues = [], []

            with torch.no_grad():
                for batch in self.loaders['val']:
                    batch = batch.to(self.device)
                    pred = self.model(batch)

                    loss_data = F.huber_loss(pred, batch.y, delta=0.5)
                    loss_physics = self._physics_loss(pred, batch)
                    loss_eps = (self.model.epsilon() ** 2) * self.lambda_int
                    loss = loss_data + physics_weight * loss_physics + loss_eps
                    val_loss += loss.item()

                    pred_unnorm = pred * self.scale + self.mean
                    true_unnorm = batch.y * self.scale + self.mean
                    val_preds.append(pred_unnorm.cpu().numpy())
                    val_trues.append(true_unnorm.cpu().numpy())

            avg_val_loss = val_loss / len(self.loaders['val'])
            self.loss_history['val'].append(avg_val_loss)

            val_preds = np.concatenate(val_preds)
            val_trues = np.concatenate(val_trues)
            val_rmse = np.sqrt(np.mean((val_preds - val_trues) ** 2))
            self.val_rmse_history.append(val_rmse)

            # ---- Test (each epoch) ----
            self.model.eval()
            test_loss = 0.0
            test_preds, test_trues = [], []

            with torch.no_grad():
                for batch in self.loaders['test']:
                    batch = batch.to(self.device)
                    pred = self.model(batch)

                    loss_data = F.huber_loss(pred, batch.y, delta=0.5)
                    loss_physics = self._physics_loss(pred, batch)
                    loss_eps = (self.model.epsilon() ** 2) * self.lambda_int
                    loss = loss_data + physics_weight * loss_physics + loss_eps
                    test_loss += loss.item()

                    pred_unnorm = pred * self.scale + self.mean
                    true_unnorm = batch.y * self.scale + self.mean
                    test_preds.append(pred_unnorm.cpu().numpy())
                    test_trues.append(true_unnorm.cpu().numpy())

            avg_test_loss = test_loss / len(self.loaders['test'])
            self.loss_history['test'].append(avg_test_loss)

            test_preds = np.concatenate(test_preds)
            test_trues = np.concatenate(test_trues)
            test_rmse = np.sqrt(np.mean((test_preds - test_trues) ** 2))
            self.test_rmse_history.append(test_rmse)

            # store for CSVs
            self.training_history_data['epoch'].append(epoch + 1)
            self.training_history_data['train_loss'].append(avg_train_loss)
            self.training_history_data['val_loss'].append(avg_val_loss)
            self.training_history_data['test_loss'].append(avg_test_loss)

            self.rmse_propagation_data['epoch'].append(epoch + 1)
            self.rmse_propagation_data['train_rmse'].append(train_rmse)
            self.rmse_propagation_data['val_rmse'].append(val_rmse)
            self.rmse_propagation_data['test_rmse'].append(test_rmse)

            # ---- GRU diagnostics: compute per-epoch means from accumulated batch outputs ----
            if len(self._train_batch_hidden_s) > 0:
                s_cat = torch.cat(self._train_batch_hidden_s, dim=0)
                l_cat = torch.cat(self._train_batch_hidden_l, dim=0)
                s_mean_norm = s_cat.norm(dim=1).mean().item()
                l_mean_norm = l_cat.norm(dim=1).mean().item()
            else:
                s_mean_norm = float('nan')
                l_mean_norm = float('nan')

            epoch_avg_pred_torsion = float(np.mean(train_preds)) if len(train_preds) else float('nan')
            self._epoch_hidden_norm_s.append(s_mean_norm)
            self._epoch_hidden_norm_l.append(l_mean_norm)
            self._epoch_avg_pred_torsion.append(epoch_avg_pred_torsion)

            # reset accumulators for next epoch
            self._train_batch_hidden_s.clear()
            self._train_batch_hidden_l.clear()

            self.scheduler.step(avg_val_loss)

            # checkpoint + early stop
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.best_model_state = self.model.state_dict().copy()
                early_stop_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                early_stop_counter += 1

            alpha, beta, gamma, eps = self.model.nonneg_weights()
            print(f'Epoch {epoch+1:03d} | '
                  f'Train Loss: {avg_train_loss:.4f} | '
                  f'Val Loss: {avg_val_loss:.4f} | '
                  f'Test Loss: {avg_test_loss:.4f} | '
                  f'Train RMSE: {train_rmse:.2f} kNm | '
                  f'Val RMSE: {val_rmse:.2f} kNm | '
                  f'Test RMSE: {test_rmse:.2f} kNm | '
                  f'Phys Wt: {physics_weight:.2f} | '
                  f'α:{alpha.item():.3f} β:{beta.item():.3f} γ:{gamma.item():.3f} ε:{eps.item():.5f} | '
                  f'LR: {self.optimizer.param_groups[0]["lr"]:.2e}')

            if early_stop_counter >= 30:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        torch.save(self.model.state_dict(), 'final_model.pth')
        print(f"Training completed in {time.time() - start_time:.2f} seconds")

    def evaluate(self):
        self.model.eval()
        results = {}
        with torch.no_grad():
            for phase in ['train', 'val', 'test']:
                preds, trues, indices = [], [], []
                for batch in self.loaders[phase]:
                    batch = batch.to(self.device)
                    pred = self.model(batch).cpu().numpy()
                    true = batch.y.cpu().numpy()

                    indices.extend(batch.idx.cpu().numpy())

                    # unnormalize
                    pred = pred * self.dataset.scalers['target'].scale_ + self.dataset.scalers['target'].mean_
                    true = true * self.dataset.scalers['target'].scale_ + self.dataset.scalers['target'].mean_

                    preds.append(pred)
                    trues.append(true)

                preds = np.concatenate(preds)
                trues = np.concatenate(trues)
                indices = np.array(indices)

                order = np.argsort(indices)
                preds = preds[order]
                trues = trues[order]

                mse = np.mean((preds - trues) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(preds - trues))
                r2 = r2_score(trues, preds)

                results[phase] = {
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'predictions': preds,
                    'targets': trues,
                    'residuals': trues - preds,
                    'indices': indices[order]
                }
        return results

    def save_predictions(self):
        results = self.evaluate()
        for phase in ['train', 'val', 'test']:
            df = self.dataset.datasets[phase]['original'].iloc[results[phase]['indices']].copy()
            df['Ts_pred'] = results[phase]['predictions']
            df['residual'] = results[phase]['residuals']
            df.to_csv(f'{phase}_predictions.csv', index=False)
            print(f"Saved {len(df)} {phase} predictions to {phase}_predictions.csv")

    def save_model_metrics(self):
        results = self.evaluate()
        metrics_data = []
        for phase in ['train', 'val', 'test']:
            metrics_data.append({
                'Phase': phase,
                'RMSE': results[phase]['RMSE'],
                'MAE': results[phase]['MAE'],
                'R2': results[phase]['R2']
            })
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv('model_metrics.csv', index=False)
        print("Saved model metrics to model_metrics.csv")
        return metrics_df

    def save_combined_predictions(self):
        results = self.evaluate()
        combined_data = []
        for phase in ['train', 'val', 'test']:
            df = self.dataset.datasets[phase]['original'].iloc[results[phase]['indices']].copy()
            df['Ts_pred'] = results[phase]['predictions']
            df['residual'] = results[phase]['residuals']
            df['phase'] = phase
            combined_data.append(df)
        combined_df = pd.concat(combined_data, ignore_index=True)
        combined_df.to_csv('combined_predictions.csv', index=False)
        print(f"Saved {len(combined_df)} combined predictions to combined_predictions.csv")
        return combined_df

    def save_graph_data(self):
        # training history
        history_df = pd.DataFrame({
            'epoch': self.training_history_data['epoch'],
            'train_loss': self.training_history_data['train_loss'],
            'val_loss': self.training_history_data['val_loss'],
            'test_loss': self.training_history_data['test_loss']
        })
        history_df.to_csv('training_history_data.csv', index=False)

        # RMSE propagation
        rmse_df = pd.DataFrame({
            'epoch': self.rmse_propagation_data['epoch'],
            'train_rmse': self.rmse_propagation_data['train_rmse'],
            'val_rmse': self.rmse_propagation_data['val_rmse'],
            'test_rmse': self.rmse_propagation_data['test_rmse']
        })
        rmse_df.to_csv('rmse_propagation_data.csv', index=False)

        # residual analysis per phase
        results = self.evaluate()
        for phase in ['train', 'val', 'test']:
            phase_data = results[phase]
            residual_df = pd.DataFrame({
                'actual': phase_data['targets'],
                'predicted': phase_data['predictions'],
                'residual': phase_data['residuals']
            })
            residual_df.to_csv(f'{phase}_residual_analysis_data.csv', index=False)

        # QQ plot data
        for phase in ['train', 'val', 'test']:
            residuals = results[phase]['residuals']
            residuals_sorted = np.sort(residuals)
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals_sorted)))
            qq_df = pd.DataFrame({
                'theoretical_quantiles': theoretical_quantiles,
                'ordered_residuals': residuals_sorted
            })
            qq_df.to_csv(f'{phase}_qq_plot_data.csv', index=False)

        print("Saved graph data to CSV files")

    def plot_results(self):
        # Matplotlib config
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.linewidth'] = 1.5
        plt.rcParams['lines.linewidth'] = 2.0
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['figure.titlesize'] = 18
        plt.rcParams['figure.dpi'] = 300

        metrics = self.evaluate()
        phases = ['train', 'val', 'test']

        # 1) Training history
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history['train'], label='Train', linewidth=2.5)
        plt.plot(self.loss_history['val'], label='Validation', linewidth=2.5)
        plt.plot(self.loss_history['test'], label='Test', linewidth=2.5)
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Training History')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout(); plt.savefig('training_history.png'); plt.close()

        # 2) Pred vs Actual (per phase)
        plt.figure(figsize=(15, 5))
        for i, phase in enumerate(phases):
            plt.subplot(1, 3, i+1)
            plt.scatter(metrics[phase]['targets'], metrics[phase]['predictions'],
                        alpha=0.6, s=30, color='#1f77b4', edgecolor='k')
            min_val = min(metrics[phase]['targets'].min(), metrics[phase]['predictions'].min())
            max_val = max(metrics[phase]['targets'].max(), metrics[phase]['predictions'].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            plt.xlabel('Actual Torsion (kNm)'); plt.ylabel('Predicted (kNm)')
            plt.title(f'{phase.capitalize()} Set: R²={metrics[phase]["R2"]:.3f}')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout(); plt.savefig('predictions_vs_actuals.png'); plt.close()

        # 3) RMSE propagation
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_rmse_history, label='Training RMSE', linewidth=2.5, alpha=0.8)
        plt.plot(self.val_rmse_history, label='Validation RMSE', linewidth=2.5, alpha=0.8)
        plt.plot(self.test_rmse_history, label='Test RMSE', linewidth=2.5, alpha=0.8)
        plt.xlabel('Epoch'); plt.ylabel('RMSE (kNm)'); plt.title('RMSE Propagation During Training')
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout(); plt.savefig('rmse_propagation.png'); plt.close()

        # 4) Combined prediction plot
        plt.figure(figsize=(8, 6))
        colors = {'train': 'red', 'val': 'green', 'test': 'blue'}
        markers = {'train': 'o', 'val': 's', 'test': '^'}
        marker_size = 150
        for phase in phases:
            preds = metrics[phase]['predictions']
            trues = metrics[phase]['targets']
            plt.scatter(trues, preds, c=colors[phase], marker=markers[phase],
                        alpha=0.7, s=marker_size, edgecolors='k', linewidths=0.5,
                        label=f'{phase.capitalize()} (R²={metrics[phase]["R2"]:.3f})')
        all_trues = np.concatenate([metrics[p]['targets'] for p in phases])
        all_preds = np.concatenate([metrics[p]['predictions'] for p in phases])
        min_val = min(all_trues.min(), all_preds.min())
        max_val = max(all_trues.max(), all_preds.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')
        plt.xlabel('Actual Torsion (kNm)'); plt.ylabel('Predicted Torsion (kNm)')
        plt.title('Combined Prediction Performance'); plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout(); plt.savefig('combined_predictions.png', dpi=300); plt.close()

    def plot_residual_analysis_all_phases(self):
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 12,
            'axes.linewidth': 1.5,
            'lines.linewidth': 2.0
        })

        metrics = self.evaluate()
        phases = ['train', 'val', 'test']
        colors = {'train': 'blue', 'val': 'green', 'test': 'red'}

        fig, axs = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Residual Analysis for All Phases', fontsize=16, fontweight='bold')

        for i, phase in enumerate(phases):
            phase_preds = metrics[phase]['predictions']
            phase_trues = metrics[phase]['targets']
            residuals = metrics[phase]['residuals']

            # 1) Residuals vs Predicted
            sns.scatterplot(x=phase_preds, y=residuals, ax=axs[i, 0], alpha=0.7,
                            color=colors[phase], edgecolor='k', s=150)
            axs[i, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
            axs[i, 0].set_xlabel('Predicted Torsion (kNm)'); axs[i, 0].set_ylabel('Residuals (kNm)')
            axs[i, 0].set_title(f'{phase.capitalize()} - Residuals vs Predicted')
            axs[i, 0].grid(True, linestyle='--', alpha=0.7); axs[i, 0].tick_params(labelsize=12)

            # 2) Residuals vs Actual
            sns.scatterplot(x=phase_trues, y=residuals, ax=axs[i, 1], alpha=0.7,
                            color=colors[phase], edgecolor='k', s=150)
            axs[i, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
            axs[i, 1].set_xlabel('Actual Torsion (kNm)'); axs[i, 1].set_ylabel('Residuals (kNm)')
            axs[i, 1].set_title(f'{phase.capitalize()} - Residuals vs Actual')
            axs[i, 1].grid(True, linestyle='--', alpha=0.7); axs[i, 1].tick_params(labelsize=12)

            # 3) QQ plot
            stats.probplot(residuals, dist="norm", plot=axs[i, 2])
            axs[i, 2].get_lines()[0].set_markersize(15.0)
            axs[i, 2].get_lines()[0].set_markerfacecolor(colors[phase])
            axs[i, 2].get_lines()[0].set_markeredgecolor('k')
            axs[i, 2].get_lines()[1].set_color('r'); axs[i, 2].get_lines()[1].set_linewidth(3.0)
            axs[i, 2].set_title(f'{phase.capitalize()} - QQ Plot')
            axs[i, 2].set_xlabel('Theoretical Quantiles'); axs[i, 2].set_ylabel('Ordered Residuals')
            axs[i, 2].grid(True, linestyle='--', alpha=0.7); axs[i, 2].tick_params(labelsize=12)

        plt.tight_layout(); plt.subplots_adjust(top=0.93)
        plt.savefig('residual_analysis_all_phases.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Print residual stats
        print("\nResidual Analysis Statistics for All Phases:")
        print(f"{'Phase':<10} | {'Mean':<10} | {'Std Dev':<10} | {'Skewness':<10} | {'Kurtosis':<10}")
        print("-" * 60)
        for phase in phases:
            residuals = metrics[phase]['residuals']
            residual_mean = np.mean(residuals)
            residual_std = np.std(residuals)
            residual_skew = stats.skew(residuals)
            residual_kurtosis = stats.kurtosis(residuals)
            print(f"{phase.capitalize():<10} | {residual_mean:<10.4f} | {residual_std:<10.4f} | "
                  f"{residual_skew:<10.4f} | {residual_kurtosis:<10.4f}")

            jb_test = stats.jarque_bera(residuals)
            ad_test = stats.anderson(residuals, dist='norm')
            print(f"  Jarque-Bera test: statistic={jb_test[0]:.4f}, p-value={jb_test[1]:.4f}")
            print("  Anderson-Darling test:")
            print(f"    Statistic: {ad_test.statistic:.4f}")
            for j in range(len(ad_test.critical_values)):
                sl, cv = ad_test.significance_level[j], ad_test.critical_values[j]
                print(f"    Critical value ({sl}%): {cv:.4f} - {'Normal' if ad_test.statistic < cv else 'Non-normal'}")

    def save_residual_analysis_csv(self):
        results = self.evaluate()

        # Row-wise residuals
        residual_data = []
        for phase in ['train', 'val', 'test']:
            phase_data = results[phase]
            for i in range(len(phase_data['targets'])):
                residual_data.append({
                    'phase': phase,
                    'actual': phase_data['targets'][i],
                    'predicted': phase_data['predictions'][i],
                    'residual': phase_data['residuals'][i],
                    'absolute_error': abs(phase_data['residuals'][i]),
                    'squared_error': phase_data['residuals'][i] ** 2
                })
        residual_df = pd.DataFrame(residual_data)
        residual_df.to_csv('comprehensive_residual_analysis.csv', index=False)
        print("Saved comprehensive residual analysis to comprehensive_residual_analysis.csv")

        # Summary stats
        summary_stats = residual_df.groupby('phase').agg({
            'residual': ['mean', 'std', 'min', 'max'],
            'absolute_error': 'mean',
            'squared_error': 'mean'
        }).round(4)
        summary_stats.columns = ['residual_mean', 'residual_std', 'residual_min', 'residual_max', 'mae', 'mse']
        summary_stats['rmse'] = np.sqrt(summary_stats['mse'])
        summary_stats.to_csv('residual_summary_statistics.csv')
        print("Saved residual summary statistics to residual_summary_statistics.csv")
        return residual_df, summary_stats

    def plot_gru_diagnostics(self):
        """
        Produces:
          1) Hidden state norms vs epochs (stirrup & longitudinal, last GNN layer)
          2) GRU vs Training Cycles: overlay mean predicted torsion per epoch
          3) Hidden state norm vs Predicted torsion (scatter) on TEST set
        """
        # ---------- 1) Hidden state norms vs epochs ----------
        epochs = np.arange(1, len(self._epoch_hidden_norm_s) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self._epoch_hidden_norm_s, label='Stirrup GRU hidden norm', linewidth=2.2)
        plt.plot(epochs, self._epoch_hidden_norm_l, label='Longitudinal GRU hidden norm', linewidth=2.2)
        plt.xlabel('Epoch'); plt.ylabel('Mean hidden-state L2 norm')
        plt.title('GRU Hidden-State Norms vs Epochs (last GNN layer)')
        plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
        plt.tight_layout(); plt.savefig('gru_hidden_norms_over_epochs.png', dpi=300); plt.close()

        # ---------- 2) GRU vs Training Cycles (overlay mean torsion) ----------
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(epochs, self._epoch_hidden_norm_s, label='Stirrup hidden norm', linewidth=2.0)
        ax1.plot(epochs, self._epoch_hidden_norm_l, label='Longitudinal hidden norm', linewidth=2.0)
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Mean hidden-state L2 norm')
        ax1.grid(True, linestyle='--', alpha=0.7)

        ax2 = ax1.twinx()
        ax2.plot(epochs, self._epoch_avg_pred_torsion, label='Mean predicted torsion (kN·m)', linewidth=2.0, linestyle='--')
        ax2.set_ylabel('Predicted torsion (kN·m)')

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='best')

        plt.title('GRU vs Training Cycles (Epochs)')
        fig.tight_layout(); plt.savefig('gru_vs_training_cycles.png', dpi=300); plt.close()

        # ---------- 3) Hidden state norms vs predicted torsion (TEST scatter) ----------
        self.model.eval()
        hidden_norms = []
        preds_knm = []

        with torch.no_grad():
            for batch in self.loaders['test']:
                self._train_batch_hidden_s.clear()
                self._train_batch_hidden_l.clear()

                batch = batch.to(self.device)
                pred_std = self.model(batch)
                pred_knm = (pred_std * self.scale + self.mean).cpu().numpy().reshape(-1)

                if self._train_batch_hidden_s and self._train_batch_hidden_l:
                    s_out = self._train_batch_hidden_s[-1]
                    l_out = self._train_batch_hidden_l[-1]
                    combined = 0.5 * s_out.norm(dim=1) + 0.5 * l_out.norm(dim=1)
                    hidden_norms.extend(combined.numpy().tolist())
                    preds_knm.extend(pred_knm.tolist())

                self._train_batch_hidden_s.clear()
                self._train_batch_hidden_l.clear()

        plt.figure(figsize=(8, 6))
        plt.scatter(hidden_norms, preds_knm, s=40, alpha=0.7, edgecolors='k')
        plt.xlabel('Hidden activation (mean L2 norm, last layer GRUs)')
        plt.ylabel('Predicted torsion (kN·m)')
        plt.title('Hidden State Norm vs Predicted Torsion (TEST)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout(); plt.savefig('hidden_norm_vs_predicted_torsion.png', dpi=300); plt.close()

        if len(hidden_norms) > 2:
            corr = np.corrcoef(np.asarray(hidden_norms), np.asarray(preds_knm))[0, 1]
            print(f"[Diagnostics] Pearson corr (hidden norm vs predicted torsion, TEST): {corr:.3f}")
        else:
            print("[Diagnostics] Not enough points to compute correlation.")

    def save_gru_diagnostics_csv(self):
        """
        Saves CSVs for GRU-related plots:
          - gru_over_epochs.csv : epoch, stirrup_hidden_norm, longitudinal_hidden_norm, mean_pred_torsion_knm
          - hidden_norm_vs_predicted_torsion_test.csv : hidden_norm, predicted_torsion_knm
        """
        # ---- Over-epochs CSV ----
        n = len(self._epoch_hidden_norm_s)
        if n > 0:
            df_epochs = pd.DataFrame({
                'epoch': np.arange(1, n + 1, dtype=int),
                'stirrup_hidden_norm': self._epoch_hidden_norm_s,
                'longitudinal_hidden_norm': self._epoch_hidden_norm_l,
                'mean_pred_torsion_knm': self._epoch_avg_pred_torsion
            })
            df_epochs.to_csv('gru_over_epochs.csv', index=False)
            print("Saved GRU over-epochs diagnostics to gru_over_epochs.csv")
        else:
            print("GRU diagnostics: no epoch data to save (train first).")

        # ---- Scatter CSV (recompute like in plot) ----
        self.model.eval()
        hidden_norms = []
        preds_knm = []
        with torch.no_grad():
            for batch in self.loaders['test']:
                self._train_batch_hidden_s.clear()
                self._train_batch_hidden_l.clear()

                batch = batch.to(self.device)
                pred_std = self.model(batch)
                pred_knm = (pred_std * self.scale + self.mean).cpu().numpy().reshape(-1)

                if self._train_batch_hidden_s and self._train_batch_hidden_l:
                    s_out = self._train_batch_hidden_s[-1]
                    l_out = self._train_batch_hidden_l[-1]
                    combined = 0.5 * s_out.norm(dim=1) + 0.5 * l_out.norm(dim=1)
                    hidden_norms.extend(combined.numpy().tolist())
                    preds_knm.extend(pred_knm.tolist())

                self._train_batch_hidden_s.clear()
                self._train_batch_hidden_l.clear()

        if len(hidden_norms) > 0:
            df_scatter = pd.DataFrame({
                'hidden_norm': hidden_norms,
                'predicted_torsion_knm': preds_knm
            })
            df_scatter.to_csv('hidden_norm_vs_predicted_torsion_test.csv', index=False)
            print("Saved GRU scatter diagnostics to hidden_norm_vs_predicted_torsion_test.csv")
        else:
            print("GRU diagnostics: no test scatter data to save (ensure hooks ran during a test pass).")

    def print_final_metrics(self):
        metrics = self.evaluate()
        print("\nFinal Metrics:")
        print(f"{'Phase':<8} | {'RMSE (kNm)':<12} | {'MAE (kNm)':<12} | {'R²':<8}")
        print("-" * 45)
        for phase in ['train', 'val', 'test']:
            print(f"{phase.capitalize():<8} | "
                  f"{metrics[phase]['RMSE']:<12.2f} | "
                  f"{metrics[phase]['MAE']:<12.2f} | "
                  f"{metrics[phase]['R2']:<8.3f}")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # Get data path from command-line argument
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = r"C:\Users\vikas\Desktop\torsional\mix data.csv"

    print(f"Loading data from: {data_path}")

    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}")
        print("Please check the file path and try again.")
        sys.exit(1)

    trainer = TorsionTrainer(data_path)
    trainer.train(epochs=300)
    trainer.print_final_metrics()
    trainer.plot_results()
    trainer.plot_residual_analysis_all_phases()
    trainer.save_predictions()

    # Save model metrics and combined predictions to CSV files
    trainer.save_model_metrics()
    trainer.save_combined_predictions()

    # Save residual analysis to CSV files
    trainer.save_residual_analysis_csv()

    # Save graph data to CSV files
    trainer.save_graph_data()

    # GRU diagnostics: plots + CSVs
    trainer.plot_gru_diagnostics()
    trainer.save_gru_diagnostics_csv()
