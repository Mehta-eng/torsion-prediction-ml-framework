# TorsionTrainer.py
# Full script with Physics-GNN + Overfitting/Robustness checks
import os, sys, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRUCell, LayerNorm
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP

# ---------------------------
# Dataset
# ---------------------------
class RCBeamDataset:
    def __init__(self, data_path):
        self.raw_data = pd.read_csv(
            data_path, header=None,
            names=['b','h','bp','hp','fc','ALfyL','Atfyt/s','Ts']
        ).astype(np.float32)

        # Derived features aligned with physics
        self.raw_data['A0'] = (0.85 * self.raw_data['bp'] * self.raw_data['hp']).astype(np.float32)
        self.raw_data['ph'] = (2 * (self.raw_data['bp'] + self.raw_data['hp'])).astype(np.float32)
        self.raw_data['sqrt_fc'] = np.sqrt(self.raw_data['fc']).astype(np.float32)

        # Node-wise feature groups
        self.feature_groups = {
            'stirrup': ["Atfyt/s", "bp", "hp", "A0", "ph"],
            'longitudinal': ["ALfyL"],
            'concrete': ["fc", "b", "h", "sqrt_fc"]
        }

        # Splits
        train_val, test = train_test_split(self.raw_data, test_size=0.2, random_state=42)
        train, val = train_test_split(train_val, test_size=0.125, random_state=42)

        # Scalers
        self.scalers = {
            'stirrup': StandardScaler().fit(train[self.feature_groups['stirrup']]),
            'longitudinal': StandardScaler().fit(train[self.feature_groups['longitudinal']]),
            'concrete': StandardScaler().fit(train[self.feature_groups['concrete']]),
            'target': StandardScaler().fit(train[['Ts']])
        }

        self.datasets = {
            'train': self._normalize(train),
            'val': self._normalize(val),
            'test': self._normalize(test)
        }

    def _normalize(self, df):
        return {
            'stirrup': self.scalers['stirrup'].transform(df[self.feature_groups['stirrup']]).astype(np.float32),
            'longitudinal': self.scalers['longitudinal'].transform(df[self.feature_groups['longitudinal']]).astype(np.float32),
            'concrete': self.scalers['concrete'].transform(df[self.feature_groups['concrete']]).astype(np.float32),
            'target': self.scalers['target'].transform(df[['Ts']]).astype(np.float32),
            'original': df.copy().astype(np.float32)
        }

    def _create_graph(self, idx, phase):
        d = HeteroData()
        for node_type in ['stirrup','longitudinal','concrete']:
            if node_type == 'stirrup':
                cols = ["Atfyt/s","bp","hp","A0","ph"]
            elif node_type == 'longitudinal':
                cols = ["ALfyL"]
            else:
                cols = ["fc","b","h","sqrt_fc"]

            x = self.datasets[phase][node_type][idx].reshape(1, -1)
            d[node_type].x = torch.tensor(x, dtype=torch.float32)

            orig = self.datasets[phase]['original'].iloc[idx][cols].values.reshape(1, -1)
            d[node_type].orig = torch.tensor(orig, dtype=torch.float32)

            d[node_type].edge_index = torch.tensor([[0],[0]], dtype=torch.long)

        d['stirrup','confines','longitudinal'].edge_index = torch.tensor([[0],[0]])
        d['longitudinal','transfers','concrete'].edge_index = torch.tensor([[0],[0]])
        d['stirrup','confines','concrete'].edge_index = torch.tensor([[0],[0]])

        d.y = torch.tensor(self.datasets[phase]['target'][idx], dtype=torch.float32)
        d.idx = idx; d.phase = phase
        return d

    def get_loaders(self, batch_size=32):
        loaders = {}
        for phase in ['train','val','test']:
            items = [self._create_graph(i, phase) for i in range(len(self.datasets[phase]['target']))]
            loaders[phase] = DataLoader(items, batch_size=batch_size, shuffle=(phase=='train'),
                                        follow_batch=['stirrup','longitudinal','concrete'])
        return loaders

# ---------------------------
# GNN Layers
# ---------------------------
class EnhancedGNNLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.msg_s2l = MLP([2*hidden_dim, 2*hidden_dim, hidden_dim], norm="batch_norm")
        self.msg_l2c = MLP([2*hidden_dim, 2*hidden_dim, hidden_dim], norm="batch_norm")
        self.msg_s2c = MLP([2*hidden_dim, 2*hidden_dim, hidden_dim], norm="batch_norm")

        self.gru_s = GRUCell(hidden_dim, hidden_dim)
        self.gru_l = GRUCell(hidden_dim, hidden_dim)
        self.mlp_upd_c = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )

        self.norm_s = LayerNorm(hidden_dim)
        self.norm_l = LayerNorm(hidden_dim)
        self.norm_c = LayerNorm(hidden_dim)

    def forward(self, xdict, eidx):
        h_s, h_l, h_c = xdict['stirrup'], xdict['longitudinal'], xdict['concrete']
        a_s = torch.zeros_like(h_s); a_l = torch.zeros_like(h_l); a_c = torch.zeros_like(h_c)

        key = ('stirrup','confines','longitudinal')
        if key in eidx:
            ei = eidx[key]
            m = self.msg_s2l(torch.cat([h_s[ei[0]], h_l[ei[1]]], dim=-1))
            a_l = a_l.index_add(0, ei[1], m)

        key = ('longitudinal','transfers','concrete')
        if key in eidx:
            ei = eidx[key]
            m = self.msg_l2c(torch.cat([h_l[ei[0]], h_c[ei[1]]], dim=-1))
            a_c = a_c.index_add(0, ei[1], m)

        key = ('stirrup','confines','concrete')
        if key in eidx:
            ei = eidx[key]
            m = self.msg_s2c(torch.cat([h_s[ei[0]], h_c[ei[1]]], dim=-1))
            a_c = a_c.index_add(0, ei[1], m)

        h_s_new = self.gru_s(a_s, h_s); h_s_out = self.norm_s(h_s + h_s_new)
        h_l_new = self.gru_l(a_l, h_l); h_l_out = self.norm_l(h_l + h_l_new)
        h_c_new = self.mlp_upd_c(torch.cat([h_c, a_c], dim=-1)); h_c_out = self.norm_c(h_c + h_c_new)

        return {'stirrup': h_s_out, 'longitudinal': h_l_out, 'concrete': h_c_out}

class PhysicsGNN(nn.Module):
    def __init__(self, hidden_dim=512, num_layers=6, dropout=0.3):
        super().__init__()
        self.encoders = nn.ModuleDict({
            'stirrup': nn.Sequential(MLP([5, hidden_dim, hidden_dim], norm="batch_norm"), nn.ReLU(), nn.Dropout(dropout)),
            'longitudinal': nn.Sequential(MLP([1, hidden_dim, hidden_dim], norm="batch_norm"), nn.ReLU(), nn.Dropout(dropout)),
            'concrete': nn.Sequential(MLP([4, hidden_dim, hidden_dim], norm="batch_norm"), nn.ReLU(), nn.Dropout(dropout))
        })
        self.layers = nn.ModuleList([EnhancedGNNLayer(hidden_dim) for _ in range(num_layers)])

        self.t_s = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 1))
        self.t_l = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 1))
        self.t_c = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 1))
        self.t_int = nn.Sequential(nn.Linear(3*hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_beta  = nn.Parameter(torch.tensor(0.0))
        self.log_gamma = nn.Parameter(torch.tensor(0.0))
        self.log_eps   = nn.Parameter(torch.tensor(-4.0))

    def nonneg_weights(self):
        return (F.softplus(self.log_alpha), F.softplus(self.log_beta),
                F.softplus(self.log_gamma), F.softplus(self.log_eps))

    def epsilon(self):
        return self.nonneg_weights()[-1]

    def forward(self, data):
        h = {
            'stirrup': self.encoders['stirrup'](data['stirrup'].x),
            'longitudinal': self.encoders['longitudinal'](data['longitudinal'].x),
            'concrete': self.encoders['concrete'](data['concrete'].x)
        }
        for layer in self.layers:
            h = layer(h, data.edge_index_dict)

        t_s = self.t_s(h['stirrup']).squeeze(-1)
        t_l = self.t_l(h['longitudinal']).squeeze(-1)
        t_c = self.t_c(h['concrete']).squeeze(-1)
        t_int = self.t_int(torch.cat([h['stirrup'], h['longitudinal'], h['concrete']], dim=-1)).squeeze(-1)

        alpha, beta, gamma, eps = self.nonneg_weights()
        T_pred = alpha*t_c + beta*t_s + gamma*t_l + eps*t_int
        return T_pred

# ---------------------------
# Helpers for CV/Learning curves
# ---------------------------
def _make_dataset_from_dfs(df_train, df_val, df_test, feature_groups):
    scalers = {
        'stirrup': StandardScaler().fit(df_train[feature_groups['stirrup']]),
        'longitudinal': StandardScaler().fit(df_train[feature_groups['longitudinal']]),
        'concrete': StandardScaler().fit(df_train[feature_groups['concrete']]),
        'target': StandardScaler().fit(df_train[['Ts']])
    }
    def _norm(df):
        return {
            'stirrup': scalers['stirrup'].transform(df[feature_groups['stirrup']]).astype(np.float32),
            'longitudinal': scalers['longitudinal'].transform(df[feature_groups['longitudinal']]).astype(np.float32),
            'concrete': scalers['concrete'].transform(df[feature_groups['concrete']]).astype(np.float32),
            'target': scalers['target'].transform(df[['Ts']]).astype(np.float32),
            'original': df.copy().astype(np.float32)
        }
    dsets = {'train': _norm(df_train), 'val': _norm(df_val), 'test': _norm(df_test)}
    return scalers, dsets

def _make_loaders_from_datasets(dsets, batch_size=32):
    def _create_item(idx, phase, dsets=dsets):
        d = HeteroData()
        mapping = {
            'stirrup': ["Atfyt/s","bp","hp","A0","ph"],
            'longitudinal': ["ALfyL"],
            'concrete': ["fc","b","h","sqrt_fc"]
        }
        for node_type, cols in mapping.items():
            x = dsets[phase][node_type][idx].reshape(1, -1)
            d[node_type].x = torch.tensor(x, dtype=torch.float32)
            orig = dsets[phase]['original'].iloc[idx][cols].values.reshape(1, -1)
            d[node_type].orig = torch.tensor(orig, dtype=torch.float32)
            d[node_type].edge_index = torch.tensor([[0],[0]], dtype=torch.long)
        d['stirrup','confines','longitudinal'].edge_index = torch.tensor([[0],[0]])
        d['longitudinal','transfers','concrete'].edge_index = torch.tensor([[0],[0]])
        d['stirrup','confines','concrete'].edge_index   = torch.tensor([[0],[0]])
        d.y = torch.tensor(dsets[phase]['target'][idx], dtype=torch.float32)
        d.idx = idx; d.phase = phase
        return d

    loaders = {}
    for phase in ['train','val','test']:
        items = [_create_item(i, phase) for i in range(len(dsets[phase]['target']))]
        loaders[phase] = DataLoader(items, batch_size=batch_size, shuffle=(phase=='train'),
                                    follow_batch=['stirrup','longitudinal','concrete'])
    return loaders

# ---------------------------
# Overfitting/Robustness mixin
# ---------------------------
class OverfitChecksMixin:
    def learning_curve(self, train_fracs=(0.1,0.2,0.4,0.6,0.8,1.0), epochs=120, repeats=1, random_state=42):
        np.random.seed(random_state)
        full_train = self.dataset.datasets['train']['original']
        val_df     = self.dataset.datasets['val']['original']
        test_df    = self.dataset.datasets['test']['original']

        results = []
        for frac in train_fracs:
            n = max(8, int(len(full_train)*frac))
            for r in range(repeats):
                df_tr_sub = full_train.sample(n=n, random_state=random_state+r)
                scalers, dsets = _make_dataset_from_dfs(df_tr_sub, val_df, test_df, self.dataset.feature_groups)
                loaders = _make_loaders_from_datasets(dsets, batch_size=32)

                model = PhysicsGNN(hidden_dim=512, num_layers=6).to(self.device)
                opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

                best_val = np.inf; bad = 0
                for _ in range(epochs):
                    model.train()
                    for batch in loaders['train']:
                        batch = batch.to(self.device); pred = model(batch)
                        loss = F.huber_loss(pred, batch.y, delta=0.5) \
                             + 0.5*self._physics_loss(pred, batch) \
                             + (model.epsilon()**2)*self.lambda_int
                        opt.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                        opt.step()

                    # early stop on val
                    model.eval(); v_loss=0
                    with torch.no_grad():
                        for batch in loaders['val']:
                            batch = batch.to(self.device); pred = model(batch)
                            loss = F.huber_loss(pred, batch.y, delta=0.5) \
                                 + 0.5*self._physics_loss(pred, batch) \
                                 + (model.epsilon()**2)*self.lambda_int
                            v_loss += loss.item()
                    v_loss /= max(1,len(loaders['val']))
                    if v_loss < best_val: best_val=v_loss; bad=0
                    else: bad += 1
                    if bad >= 20: break

                # test
                model.eval(); P=[]; T=[]
                with torch.no_grad():
                    for batch in loaders['test']:
                        batch = batch.to(self.device); pred = model(batch)
                        p = pred.cpu().numpy()*scalers['target'].scale_ + scalers['target'].mean_
                        t = batch.y.cpu().numpy()*scalers['target'].scale_ + scalers['target'].mean_
                        P.append(p); T.append(t)
                P=np.concatenate(P); T=np.concatenate(T)
                rmse=float(np.sqrt(np.mean((P-T)**2))); mae=float(np.mean(np.abs(P-T))); r2=float(r2_score(T,P))
                results.append({'train_frac':frac,'repeat':r,'N_train':n,'RMSE':rmse,'MAE':mae,'R2':r2})

        df = pd.DataFrame(results)
        df.to_csv('learning_curve_results.csv', index=False)

        plt.figure(figsize=(7,5))
        grp = df.groupby('train_frac')
        means = grp[['RMSE','MAE','R2']].mean()
        stds  = grp[['RMSE','MAE','R2']].std()
        plt.errorbar(means.index, means['RMSE'], yerr=stds['RMSE'], marker='o', label='RMSE')
        plt.errorbar(means.index, means['MAE'], yerr=stds['MAE'], marker='s', label='MAE')
        plt.xlabel('Training fraction'); plt.ylabel('Error'); plt.title('Learning curves (error vs. data size)')
        plt.grid(True, ls='--', alpha=0.7); plt.legend(); plt.tight_layout()
        plt.savefig('learning_curve.png', dpi=300); plt.close()
        return df

    def k_fold_cv(self, k=5, epochs=200, stratify_col=None, random_state=42):
        df = self.dataset.raw_data.copy()
        if stratify_col is not None and stratify_col in df.columns:
            bins = pd.qcut(df[stratify_col].rank(method='first'), q=min(10, len(df)//5), duplicates='drop')
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
            splits = skf.split(df, bins)
        else:
            kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
            splits = kf.split(df)

        metrics_list = []
        for fold, (idx_tr, idx_te) in enumerate(splits, start=1):
            df_tr = df.iloc[idx_tr].copy()
            df_te = df.iloc[idx_te].copy()
            df_tr, df_val = train_test_split(df_tr, test_size=0.125, random_state=random_state)

            scalers, dsets = _make_dataset_from_dfs(df_tr, df_val, df_te, self.dataset.feature_groups)
            loaders = _make_loaders_from_datasets(dsets, batch_size=32)
            model = PhysicsGNN(hidden_dim=512, num_layers=6).to(self.device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

            best_val = np.inf; bad = 0
            for _ in range(epochs):
                model.train()
                for batch in loaders['train']:
                    batch = batch.to(self.device); pred = model(batch)
                    loss = F.huber_loss(pred, batch.y, delta=0.5) \
                         + 0.5*self._physics_loss(pred, batch) \
                         + (model.epsilon()**2)*self.lambda_int
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                model.eval(); v_loss=0
                with torch.no_grad():
                    for batch in loaders['val']:
                        batch = batch.to(self.device); pred = model(batch)
                        loss = F.huber_loss(pred, batch.y, delta=0.5) \
                             + 0.5*self._physics_loss(pred, batch) \
                             + (model.epsilon()**2)*self.lambda_int
                        v_loss += loss.item()
                v_loss /= max(1,len(loaders['val']))
                if v_loss < best_val: best_val=v_loss; bad=0
                else: bad += 1
                if bad >= 25: break

            model.eval(); P=[]; T=[]
            with torch.no_grad():
                for batch in loaders['test']:
                    batch = batch.to(self.device); pred = model(batch)
                    p = pred.cpu().numpy()*scalers['target'].scale_ + scalers['target'].mean_
                    t = batch.y.cpu().numpy()*scalers['target'].scale_ + scalers['target'].mean_
                    P.append(p); T.append(t)
            P=np.concatenate(P); T=np.concatenate(T)
            rmse=float(np.sqrt(np.mean((P-T)**2))); mae=float(np.mean(np.abs(P-T))); r2=float(r2_score(T,P))
            metrics_list.append({'fold':fold,'RMSE':rmse,'MAE':mae,'R2':r2})
            print(f"[CV] Fold {fold}/{k}  RMSE={rmse:.2f} kN·m | MAE={mae:.2f} kN·m | R²={r2:.3f}")

        dfm = pd.DataFrame(metrics_list)
        dfm.to_csv('kfold_cv_metrics.csv', index=False)
        mean_row = dfm[['RMSE','MAE','R2']].mean().rename('mean')
        std_row  = dfm[['RMSE','MAE','R2']].std().rename('std')
        summary = pd.concat([mean_row, std_row], axis=1)
        summary.to_csv('kfold_cv_summary.csv')
        print("\n[CV] summary (mean ± SD):")
        print(summary)
        return dfm, summary

    def y_scramble_test(self, epochs=80, repeats=5, random_state=42):
        rng = np.random.RandomState(random_state)
        results = []
        base = self.dataset.datasets
        for r in range(repeats):
            dsets = {phase:{k:(v.copy() if isinstance(v, np.ndarray) else v.copy())
                            for k,v in base[phase].items()} for phase in ['train','val','test']}
            y_tr = dsets['train']['target'].copy()
            rng.shuffle(y_tr)
            dsets['train']['target'] = y_tr

            loaders = _make_loaders_from_datasets(dsets, batch_size=32)
            model = PhysicsGNN(hidden_dim=512, num_layers=6).to(self.device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

            best_val=np.inf; bad=0
            for _ in range(epochs):
                model.train()
                for batch in loaders['train']:
                    batch = batch.to(self.device); pred = model(batch)
                    loss = F.huber_loss(pred, batch.y, delta=0.5) \
                         + 0.5*self._physics_loss(pred, batch) \
                         + (model.epsilon()**2)*self.lambda_int
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    opt.step()
                model.eval(); v_loss=0
                with torch.no_grad():
                    for batch in loaders['val']:
                        batch = batch.to(self.device); pred = model(batch)
                        loss = F.huber_loss(pred, batch.y, delta=0.5) \
                             + 0.5*self._physics_loss(pred, batch) \
                             + (model.epsilon()**2)*self.lambda_int
                        v_loss += loss.item()
                v_loss /= max(1,len(loaders['val']))
                if v_loss < best_val: best_val=v_loss; bad=0
                else: bad += 1
                if bad >= 10: break

            model.eval(); P=[]; T=[]
            with torch.no_grad():
                for batch in loaders['test']:
                    batch = batch.to(self.device); pred = model(batch)
                    p = pred.cpu().numpy()*self.dataset.scalers['target'].scale_ + self.dataset.scalers['target'].mean_
                    t = batch.y.cpu().numpy()*self.dataset.scalers['target'].scale_ + self.dataset.scalers['target'].mean_
                    P.append(p); T.append(t)
            P=np.concatenate(P); T=np.concatenate(T)
            rmse=float(np.sqrt(np.mean((P-T)**2))); mae=float(np.mean(np.abs(P-T))); r2=float(r2_score(T,P))
            results.append({'repeat':r,'RMSE':rmse,'MAE':mae,'R2':r2})
            print(f"[Y-Scramble] {r+1}/{repeats}: RMSE={rmse:.2f} kN·m | MAE={mae:.2f} kN·m | R²={r2:.3f}")

        df = pd.DataFrame(results)
        df.to_csv('y_scramble_results.csv', index=False)
        return df

    def leakage_check(self, id_cols=None):
        def _keyset(df, cols):
            if not cols:  # fallback: hash full rows
                return set(pd.util.hash_pandas_object(df, index=False).values)
            return set(pd.util.hash_pandas_object(df[cols], index=False).values)

        train_df = self.dataset.datasets['train']['original']
        val_df   = self.dataset.datasets['val']['original']
        test_df  = self.dataset.datasets['test']['original']

        S_tr = _keyset(train_df, id_cols)
        S_va = _keyset(val_df, id_cols)
        S_te = _keyset(test_df, id_cols)

        leaks = {'train∩val': len(S_tr & S_va), 'train∩test': len(S_tr & S_te), 'val∩test': len(S_va & S_te)}
        pd.Series(leaks).to_csv('leakage_check.csv')
        print("[Leakage] Potential duplicates across splits:", leaks)
        return leaks

# ---------------------------
# Trainer
# ---------------------------
class TorsionTrainer(OverfitChecksMixin):
    def __init__(self, data_path, lr=1e-4, weight_decay=1e-5, batch_size=32):
        self.dataset = RCBeamDataset(data_path)
        self.loaders = self.dataset.get_loaders(batch_size=batch_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.model = PhysicsGNN(hidden_dim=512, num_layers=6, dropout=0.3).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=15, verbose=True)

        self.loss_history = {'train': [], 'val': [], 'test': []}
        self.train_rmse_history, self.val_rmse_history, self.test_rmse_history = [], [], []
        self.best_val_loss = float('inf'); self.best_model_state = None

        self.scale = torch.tensor(self.dataset.scalers['target'].scale_.astype(np.float32), device=self.device)
        self.mean  = torch.tensor(self.dataset.scalers['target'].mean_.astype(np.float32), device=self.device)

        self.lambda_int = 1e-4  # small penalty on ε
        # diagnostics containers
        self._epoch_hidden_norm_s, self._epoch_hidden_norm_l, self._epoch_avg_pred_torsion = [], [], []
        self._train_batch_hidden_s, self._train_batch_hidden_l = [], []

        last_layer = None
        # create a dummy layer to register hooks after training starts
        # we will hook on the last layer after first forward pass in train loop

    def _physics_loss(self, pred, batch):
        fc  = batch['concrete'].orig[:, 0].float()
        bp  = batch['stirrup'].orig[:, 1].float()
        hp  = batch['stirrup'].orig[:, 2].float()
        Atf = batch['stirrup'].orig[:, 0].float()
        AL  = batch['longitudinal'].orig[:, 0].float()

        A0 = 0.85 * bp * hp
        ph = 2 * (bp + hp)
        T_conc = (0.33 * torch.sqrt(fc) * (A0**2) / ph) / 1e6
        T_st   = (2 * A0 * Atf) / 1e6
        T_long = (2 * A0 * AL / ph) / 1e6
        T_phys = T_conc + T_st + T_long
        Tn = (T_phys - self.mean) / self.scale
        return F.huber_loss(pred, Tn, delta=0.5)

    def _ensure_hooks(self):
        if hasattr(self, "_hooks_ready") and self._hooks_ready:
            return
        if len(self.model.layers) == 0:
            return
        last_layer = self.model.layers[-1]
        def _hook_s(module, inputs, output): self._train_batch_hidden_s.append(output.detach().cpu())
        def _hook_l(module, inputs, output): self._train_batch_hidden_l.append(output.detach().cpu())
        self._gru_hook_handles = [
            last_layer.gru_s.register_forward_hook(_hook_s),
            last_layer.gru_l.register_forward_hook(_hook_l),
        ]
        self._hooks_ready = True

    def train(self, epochs=300):
        start = time.time()
        early_stop_counter = 0; patience = 30

        for epoch in range(epochs):
            self._ensure_hooks()
            physics_weight = max(0.1, 1.5 - epoch / 100)

            # ---- Train ----
            self.model.train(); train_loss = 0.0; train_preds=[]; train_trues=[]
            for batch in self.loaders['train']:
                self.optimizer.zero_grad()
                batch = batch.to(self.device)
                pred = self.model(batch)

                loss = F.huber_loss(pred, batch.y, delta=0.5) \
                     + physics_weight * self._physics_loss(pred, batch) \
                     + (self.model.epsilon()**2) * self.lambda_int
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item()

                with torch.no_grad():
                    p = pred * self.scale + self.mean
                    t = batch.y * self.scale + self.mean
                    train_preds.append(p.cpu().numpy()); train_trues.append(t.cpu().numpy())

            avg_train_loss = train_loss / len(self.loaders['train'])
            self.loss_history['train'].append(avg_train_loss)
            train_preds = np.concatenate(train_preds); train_trues = np.concatenate(train_trues)
            train_rmse = float(np.sqrt(np.mean((train_preds - train_trues)**2)))
            self.train_rmse_history.append(train_rmse)

            # ---- Val ----
            self.model.eval(); val_loss=0.0; val_preds=[]; val_trues=[]
            with torch.no_grad():
                for batch in self.loaders['val']:
                    batch = batch.to(self.device)
                    pred = self.model(batch)
                    loss = F.huber_loss(pred, batch.y, delta=0.5) \
                         + physics_weight * self._physics_loss(pred, batch) \
                         + (self.model.epsilon()**2) * self.lambda_int
                    val_loss += loss.item()
                    p = pred * self.scale + self.mean
                    t = batch.y * self.scale + self.mean
                    val_preds.append(p.cpu().numpy()); val_trues.append(t.cpu().numpy())

            avg_val_loss = val_loss / len(self.loaders['val'])
            self.loss_history['val'].append(avg_val_loss)
            val_preds = np.concatenate(val_preds); val_trues = np.concatenate(val_trues)
            val_rmse = float(np.sqrt(np.mean((val_preds - val_trues)**2)))
            self.val_rmse_history.append(val_rmse)

            # ---- Test ----
            self.model.eval(); test_loss=0.0; test_preds=[]; test_trues=[]
            with torch.no_grad():
                for batch in self.loaders['test']:
                    batch = batch.to(self.device)
                    pred = self.model(batch)
                    loss = F.huber_loss(pred, batch.y, delta=0.5) \
                         + physics_weight * self._physics_loss(pred, batch) \
                         + (self.model.epsilon()**2) * self.lambda_int
                    test_loss += loss.item()
                    p = pred * self.scale + self.mean
                    t = batch.y * self.scale + self.mean
                    test_preds.append(p.cpu().numpy()); test_trues.append(t.cpu().numpy())

            avg_test_loss = test_loss / len(self.loaders['test'])
            self.loss_history['test'].append(avg_test_loss)
            test_preds = np.concatenate(test_preds); test_trues = np.concatenate(test_trues)
            test_rmse = float(np.sqrt(np.mean((test_preds - test_trues)**2)))
            self.test_rmse_history.append(test_rmse)

            # epoch diagnostics for GRUs
            if len(self._train_batch_hidden_s)>0:
                s_cat = torch.cat(self._train_batch_hidden_s, dim=0)
                l_cat = torch.cat(self._train_batch_hidden_l, dim=0)
                s_mean_norm = s_cat.norm(dim=1).mean().item()
                l_mean_norm = l_cat.norm(dim=1).mean().item()
            else:
                s_mean_norm = float('nan'); l_mean_norm = float('nan')
            epoch_avg_pred_torsion = float(np.mean(train_preds)) if len(train_preds) else float('nan')
            self._epoch_hidden_norm_s.append(s_mean_norm)
            self._epoch_hidden_norm_l.append(l_mean_norm)
            self._epoch_avg_pred_torsion.append(epoch_avg_pred_torsion)
            self._train_batch_hidden_s.clear(); self._train_batch_hidden_l.clear()

            self.scheduler.step(avg_val_loss)
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.best_model_state = self.model.state_dict().copy()
                torch.save(self.model.state_dict(), 'best_model.pth')
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            a,b,g,e = self.model.nonneg_weights()
            print(f"Epoch {epoch+1:03d} | "
                  f"TrainLoss {avg_train_loss:.4f} | ValLoss {avg_val_loss:.4f} | TestLoss {avg_test_loss:.4f} | "
                  f"RMSE T/V/S {train_rmse:.2f}/{val_rmse:.2f}/{test_rmse:.2f} kN·m | "
                  f"α:{a.item():.3f} β:{b.item():.3f} γ:{g.item():.3f} ε:{e.item():.5f} | "
                  f"LR {self.optimizer.param_groups[0]['lr']:.2e}")

            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        torch.save(self.model.state_dict(), 'final_model.pth')
        print(f"Training completed in {time.time()-start:.2f}s")

    def evaluate(self):
        self.model.eval()
        results = {}
        with torch.no_grad():
            for phase in ['train','val','test']:
                preds, trues, indices = [], [], []
                for batch in self.loaders[phase]:
                    batch = batch.to(self.device)
                    pred = self.model(batch).cpu().numpy()
                    true = batch.y.cpu().numpy()
                    indices.extend(batch.idx.cpu().numpy())
                    pred = pred * self.dataset.scalers['target'].scale_ + self.dataset.scalers['target'].mean_
                    true = true * self.dataset.scalers['target'].scale_ + self.dataset.scalers['target'].mean_
                    preds.append(pred); trues.append(true)

                preds = np.concatenate(preds); trues = np.concatenate(trues); indices = np.array(indices)
                order = np.argsort(indices); preds = preds[order]; trues = trues[order]
                mse = np.mean((preds-trues)**2); rmse=np.sqrt(mse); mae=np.mean(np.abs(preds-trues)); r2=r2_score(trues,preds)
                results[phase] = {'RMSE':rmse,'MAE':mae,'R2':r2,'predictions':preds,'targets':trues,
                                  'residuals':trues-preds,'indices':indices[order]}
        return results

    def print_final_metrics(self):
        metrics = self.evaluate()
        print("\nFinal Metrics:")
        print(f"{'Phase':<8} | {'RMSE (kN·m)':<12} | {'MAE (kN·m)':<12} | {'R²':<6}")
        print("-"*44)
        for phase in ['train','val','test']:
            print(f"{phase.capitalize():<8} | {metrics[phase]['RMSE']:<12.2f} | "
                  f"{metrics[phase]['MAE']:<12.2f} | {metrics[phase]['R2']:<6.3f}")

    # ---- Saving plots/data (unchanged essentials) ----
    def plot_results(self):
        plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':12,
                             'axes.linewidth':1.5,'lines.linewidth':2.0,'axes.labelsize':14,'axes.titlesize':16,
                             'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':12,'figure.titlesize':18,'figure.dpi':300})

        metrics = self.evaluate(); phases=['train','val','test']

        # Training history
        plt.figure(figsize=(10,6))
        plt.plot(self.loss_history['train'], label='Train', linewidth=2.5)
        plt.plot(self.loss_history['val'], label='Validation', linewidth=2.5)
        plt.plot(self.loss_history['test'], label='Test', linewidth=2.5)
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Training History')
        plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout(); plt.savefig('training_history.png'); plt.close()

        # Pred vs Actual (per phase)
        plt.figure(figsize=(15,5))
        for i, phase in enumerate(phases):
            plt.subplot(1,3,i+1)
            plt.scatter(metrics[phase]['targets'], metrics[phase]['predictions'], alpha=0.6, s=30, color='#1f77b4', edgecolor='k')
            miv = min(metrics[phase]['targets'].min(), metrics[phase]['predictions'].min())
            mav = max(metrics[phase]['targets'].max(), metrics[phase]['predictions'].max())
            plt.plot([miv, mav], [miv, mav], 'r--', linewidth=2)
            plt.xlabel('Actual Torsion (kN·m)'); plt.ylabel('Predicted (kN·m)')
            plt.title(f"{phase.capitalize()} Set: R²={metrics[phase]['R2']:.3f}")
            plt.grid(True, linestyle='--', alpha=0.7); plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout(); plt.savefig('predictions_vs_actuals.png'); plt.close()

        # RMSE propagation
        plt.figure(figsize=(10,6))
        plt.plot(self.train_rmse_history, label='Training RMSE', linewidth=2.5, alpha=0.8)
        plt.plot(self.val_rmse_history, label='Validation RMSE', linewidth=2.5, alpha=0.8)
        plt.plot(self.test_rmse_history, label='Test RMSE', linewidth=2.5, alpha=0.8)
        plt.xlabel('Epoch'); plt.ylabel('RMSE (kN·m)'); plt.title('RMSE Propagation During Training')
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout(); plt.savefig('rmse_propagation.png'); plt.close()

        # Combined prediction plot
        plt.figure(figsize=(8,6))
        colors={'train':'red','val':'green','test':'blue'}; markers={'train':'o','val':'s','test':'^'}
        for phase in phases:
            preds = metrics[phase]['predictions']; trues = metrics[phase]['targets']
            plt.scatter(trues, preds, c=colors[phase], marker=markers[phase], alpha=0.7, s=150, edgecolors='k', linewidths=0.5,
                        label=f'{phase.capitalize()} (R²={metrics[phase]["R2"]:.3f})')
        all_t = np.concatenate([metrics[p]['targets'] for p in phases])
        all_p = np.concatenate([metrics[p]['predictions'] for p in phases])
        miv, mav = min(all_t.min(), all_p.min()), max(all_t.max(), all_p.max())
        plt.plot([miv, mav], [miv, mav], 'k--', linewidth=2, label='Perfect Prediction')
        plt.xlabel('Actual Torsion (kN·m)'); plt.ylabel('Predicted Torsion (kN·m)')
        plt.title('Combined Prediction Performance'); plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout(); plt.savefig('combined_predictions.png', dpi=300); plt.close()

    def plot_residual_analysis_all_phases(self):
        plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':12,'axes.linewidth':1.5,'lines.linewidth':2.0})
        metrics = self.evaluate(); phases=['train','val','test']; colors={'train':'blue','val':'green','test':'red'}
        fig, axs = plt.subplots(3,3, figsize=(18,15)); fig.suptitle('Residual Analysis for All Phases', fontsize=16, fontweight='bold')

        for i, phase in enumerate(phases):
            preds = metrics[phase]['predictions']; trues = metrics[phase]['targets']; residuals = metrics[phase]['residuals']
            sns.scatterplot(x=preds, y=residuals, ax=axs[i,0], alpha=0.7, color=colors[phase], edgecolor='k', s=150)
            axs[i,0].axhline(y=0, color='r', linestyle='--', linewidth=2)
            axs[i,0].set_xlabel('Predicted (kN·m)'); axs[i,0].set_ylabel('Residual (kN·m)')
            axs[i,0].set_title(f'{phase.capitalize()} - Residual vs Predicted'); axs[i,0].grid(True, ls='--', alpha=0.7)

            sns.scatterplot(x=trues, y=residuals, ax=axs[i,1], alpha=0.7, color=colors[phase], edgecolor='k', s=150)
            axs[i,1].axhline(y=0, color='r', linestyle='--', linewidth=2)
            axs[i,1].set_xlabel('Actual (kN·m)'); axs[i,1].set_ylabel('Residual (kN·m)')
            axs[i,1].set_title(f'{phase.capitalize()} - Residual vs Actual'); axs[i,1].grid(True, ls='--', alpha=0.7)

            stats.probplot(residuals, dist="norm", plot=axs[i,2])
            axs[i,2].get_lines()[0].set_markersize(15.0)
            axs[i,2].get_lines()[0].set_markerfacecolor(colors[phase])
            axs[i,2].get_lines()[0].set_markeredgecolor('k')
            axs[i,2].get_lines()[1].set_color('r'); axs[i,2].get_lines()[1].set_linewidth(3.0)
            axs[i,2].set_title(f'{phase.capitalize()} - QQ Plot')
            axs[i,2].set_xlabel('Theoretical Quantiles'); axs[i,2].set_ylabel('Ordered Residuals')
            axs[i,2].grid(True, ls='--', alpha=0.7)

        plt.tight_layout(); plt.subplots_adjust(top=0.93)
        plt.savefig('residual_analysis_all_phases.png', dpi=300, bbox_inches='tight'); plt.close()

        print("\nResidual Analysis Statistics for All Phases:")
        print(f"{'Phase':<10} | {'Mean':<10} | {'Std Dev':<10} | {'Skewness':<10} | {'Kurtosis':<10}")
        print("-"*60)
        for phase in phases:
            residuals = metrics[phase]['residuals']
            print(f"{phase.capitalize():<10} | {np.mean(residuals):<10.4f} | {np.std(residuals):<10.4f} | "
                  f"{stats.skew(residuals):<10.4f} | {stats.kurtosis(residuals):<10.4f}")

    # CSV exporters (kept from your version, shortened where possible)
    def save_predictions(self):
        results = self.evaluate()
        for phase in ['train','val','test']:
            df = self.dataset.datasets[phase]['original'].iloc[results[phase]['indices']].copy()
            df['Ts_pred'] = results[phase]['predictions']; df['residual'] = results[phase]['residuals']
            df.to_csv(f'{phase}_predictions.csv', index=False)
            print(f"Saved {len(df)} {phase} predictions to {phase}_predictions.csv")

    def save_model_metrics(self):
        results = self.evaluate()
        rows = [{'Phase':ph, 'RMSE':results[ph]['RMSE'], 'MAE':results[ph]['MAE'], 'R2':results[ph]['R2']} for ph in ['train','val','test']]
        df = pd.DataFrame(rows); df.to_csv('model_metrics.csv', index=False)
        print("Saved model metrics to model_metrics.csv"); return df

    def save_combined_predictions(self):
        results = self.evaluate(); parts=[]
        for phase in ['train','val','test']:
            df = self.dataset.datasets[phase]['original'].iloc[results[phase]['indices']].copy()
            df['Ts_pred'] = results[phase]['predictions']; df['residual'] = results[phase]['residuals']; df['phase']=phase
            parts.append(df)
        comb = pd.concat(parts, ignore_index=True); comb.to_csv('combined_predictions.csv', index=False)
        print(f"Saved {len(comb)} combined predictions to combined_predictions.csv"); return comb

    def save_graph_data(self):
        history_df = pd.DataFrame({'epoch':range(1, len(self.loss_history['train'])+1),
                                   'train_loss':self.loss_history['train'],
                                   'val_loss':self.loss_history['val'],
                                   'test_loss':self.loss_history['test']})
        history_df.to_csv('training_history_data.csv', index=False)

        rmse_df = pd.DataFrame({'epoch':range(1, len(self.train_rmse_history)+1),
                                'train_rmse':self.train_rmse_history,
                                'val_rmse':self.val_rmse_history,
                                'test_rmse':self.test_rmse_history})
        rmse_df.to_csv('rmse_propagation_data.csv', index=False)

        results = self.evaluate()
        for phase in ['train','val','test']:
            pd.DataFrame({'actual':results[phase]['targets'],
                          'predicted':results[phase]['predictions'],
                          'residual':results[phase]['residuals']}).to_csv(f'{phase}_residual_analysis_data.csv', index=False)

        for phase in ['train','val','test']:
            residuals = results[phase]['residuals']
            residuals_sorted = np.sort(residuals)
            theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals_sorted)))
            pd.DataFrame({'theoretical_quantiles':theoretical,'ordered_residuals':residuals_sorted}).to_csv(f'{phase}_qq_plot_data.csv', index=False)
        print("Saved graph data to CSV files")

    def save_residual_analysis_csv(self):
        results = self.evaluate()
        rows = []
        for phase in ['train','val','test']:
            r = results[phase]
            for i in range(len(r['targets'])):
                rows.append({'phase':phase,'actual':r['targets'][i],'predicted':r['predictions'][i],
                             'residual':r['residuals'][i],'absolute_error':abs(r['residuals'][i]),
                             'squared_error':r['residuals'][i]**2})
        df = pd.DataFrame(rows); df.to_csv('comprehensive_residual_analysis.csv', index=False)
        summ = df.groupby('phase').agg({'residual':['mean','std','min','max'],
                                        'absolute_error':'mean','squared_error':'mean'}).round(4)
        summ.columns = ['residual_mean','residual_std','residual_min','residual_max','mae','mse']
        summ['rmse'] = np.sqrt(summ['mse']); summ.to_csv('residual_summary_statistics.csv')
        print("Saved residual analysis to CSVs"); return df, summ

    def plot_gru_diagnostics(self):
        epochs = np.arange(1, len(self._epoch_hidden_norm_s) + 1)
        # 1) hidden norms vs epochs
        plt.figure(figsize=(10,6))
        plt.plot(epochs, self._epoch_hidden_norm_s, label='Stirrup GRU hidden norm', linewidth=2.2)
        plt.plot(epochs, self._epoch_hidden_norm_l, label='Longitudinal GRU hidden norm', linewidth=2.2)
        plt.xlabel('Epoch'); plt.ylabel('Mean hidden-state L2 norm')
        plt.title('GRU Hidden-State Norms vs Epochs (last GNN layer)')
        plt.grid(True, ls='--', alpha=0.7); plt.legend()
        plt.tight_layout(); plt.savefig('gru_hidden_norms_over_epochs.png', dpi=300); plt.close()

        # 2) overlay mean torsion
        fig, ax1 = plt.subplots(figsize=(10,6))
        ax1.plot(epochs, self._epoch_hidden_norm_s, label='Stirrup hidden norm', linewidth=2.0)
        ax1.plot(epochs, self._epoch_hidden_norm_l, label='Longitudinal hidden norm', linewidth=2.0)
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Mean hidden-state L2 norm'); ax1.grid(True, ls='--', alpha=0.7)
        ax2 = ax1.twinx()
        ax2.plot(epochs, self._epoch_avg_pred_torsion, label='Mean predicted torsion (kN·m)', linewidth=2.0, ls='--')
        ax2.set_ylabel('Predicted torsion (kN·m)')
        lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines+lines2, labels+labels2, loc='best')
        plt.title('GRU vs Training Cycles (Epochs)')
        fig.tight_layout(); plt.savefig('gru_vs_training_cycles.png', dpi=300); plt.close()

        # 3) hidden norm vs predicted torsion on TEST
        self.model.eval()
        hidden_norms, preds_knm = [], []
        with torch.no_grad():
            for batch in self.loaders['test']:
                self._train_batch_hidden_s.clear(); self._train_batch_hidden_l.clear()
                batch = batch.to(self.device); pred_std = self.model(batch)
                pred_knm = (pred_std * self.scale + self.mean).cpu().numpy().reshape(-1)
                if self._train_batch_hidden_s and self._train_batch_hidden_l:
                    s_out = self._train_batch_hidden_s[-1]; l_out = self._train_batch_hidden_l[-1]
                    combined = 0.5 * s_out.norm(dim=1) + 0.5 * l_out.norm(dim=1)
                    hidden_norms.extend(combined.numpy().tolist()); preds_knm.extend(pred_knm.tolist())
                self._train_batch_hidden_s.clear(); self._train_batch_hidden_l.clear()

        plt.figure(figsize=(8,6))
        plt.scatter(hidden_norms, preds_knm, s=40, alpha=0.7, edgecolors='k')
        plt.xlabel('Hidden activation (mean L2 norm, last layer GRUs)'); plt.ylabel('Predicted torsion (kN·m)')
        plt.title('Hidden State Norm vs Predicted Torsion (TEST)')
        plt.grid(True, ls='--', alpha=0.7); plt.tight_layout(); plt.savefig('hidden_norm_vs_predicted_torsion.png', dpi=300); plt.close()
        if len(hidden_norms) > 2:
            corr = np.corrcoef(np.asarray(hidden_norms), np.asarray(preds_knm))[0,1]
            print(f"[Diagnostics] Pearson corr (hidden norm vs predicted torsion, TEST): {corr:.3f}")
        else:
            print("[Diagnostics] Not enough points to compute correlation.")

    def save_gru_diagnostics_csv(self):
        n = len(self._epoch_hidden_norm_s)
        if n>0:
            pd.DataFrame({'epoch':np.arange(1,n+1,dtype=int),
                          'stirrup_hidden_norm':self._epoch_hidden_norm_s,
                          'longitudinal_hidden_norm':self._epoch_hidden_norm_l,
                          'mean_pred_torsion_knm':self._epoch_avg_pred_torsion}).to_csv('gru_over_epochs.csv', index=False)
            print("Saved GRU over-epochs diagnostics to gru_over_epochs.csv")
        else:
            print("GRU diagnostics: no epoch data to save (train first).")

        # scatter CSV
        self.model.eval(); hidden_norms=[]; preds_knm=[]
        with torch.no_grad():
            for batch in self.loaders['test']:
                self._train_batch_hidden_s.clear(); self._train_batch_hidden_l.clear()
                batch = batch.to(self.device); pred_std = self.model(batch)
                pred_knm = (pred_std * self.scale + self.mean).cpu().numpy().reshape(-1)
                if self._train_batch_hidden_s and self._train_batch_hidden_l:
                    s_out = self._train_batch_hidden_s[-1]; l_out = self._train_batch_hidden_l[-1]
                    combined = 0.5*s_out.norm(dim=1)+0.5*l_out.norm(dim=1)
                    hidden_norms.extend(combined.numpy().tolist()); preds_knm.extend(pred_knm.tolist())
                self._train_batch_hidden_s.clear(); self._train_batch_hidden_l.clear()

        if len(hidden_norms)>0:
            pd.DataFrame({'hidden_norm':hidden_norms,'predicted_torsion_knm':preds_knm}).to_csv('hidden_norm_vs_predicted_torsion_test.csv', index=False)
            print("Saved GRU scatter diagnostics to hidden_norm_vs_predicted_torsion_test.csv")
        else:
            print("GRU diagnostics: no test scatter data to save (ensure hooks ran during a test pass).")

# ---------------------------
# Main
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Physics-GNN for RC torsion with overfitting checks")
    p.add_argument("csv_path", help="Path to CSV data file")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    # optional checks
    p.add_argument("--run-cv", action="store_true", help="Run 5-fold cross-validation (mean ± SD)")
    p.add_argument("--run-learning-curve", action="store_true", help="Run learning curves")
    p.add_argument("--run-y-scramble", action="store_true", help="Run y-scramble stress test")
    p.add_argument("--leakage-check", action="store_true", help="Run leakage/duplication check")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    data_path = args.csv_path
    if not os.path.exists(data_path):
        print(f"Error: file not found at {data_path}"); sys.exit(1)

    trainer = TorsionTrainer(data_path, lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size)
    trainer.train(epochs=args.epochs)
    trainer.print_final_metrics()
    trainer.plot_results()
    trainer.plot_residual_analysis_all_phases()
    trainer.save_predictions()
    trainer.save_model_metrics()
    trainer.save_combined_predictions()
    trainer.save_residual_analysis_csv()
    trainer.save_graph_data()
    trainer.plot_gru_diagnostics()
    trainer.save_gru_diagnostics_csv()

    # Optional robustness checks
    if args.run_cv:
        trainer.k_fold_cv(k=5, epochs=max(120, args.epochs//2), stratify_col=None)

    if args.run_learning_curve:
        trainer.learning_curve(train_fracs=(0.1,0.2,0.4,0.6,0.8,1.0), epochs=max(80, args.epochs//3), repeats=1)

    if args.run_y_scramble:
        trainer.y_scramble_test(epochs=80, repeats=5)

    if args.leakage_check:
        trainer.leakage_check(id_cols=None)  # set to ['study_id'] if you have such a column
