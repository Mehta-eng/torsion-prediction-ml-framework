import os
import io
import re
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from torch.nn import GRUCell, LayerNorm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP

# =========================
# UI CONFIG
# =========================
st.set_page_config(page_title="Torsion GNN", layout="wide")
st.title("Physics-Guided GNN — Torsional Strength (kN·m)")

# =========================
# Feature schema
# =========================
FEATURE_GROUPS = {
    'stirrup': ["Atfyt/s", "bp", "hp", "A0", "ph"],
    'longitudinal': ["ALfyL"],
    'concrete': ["fc", "b", "h", "sqrt_fc"]
}
REQUIRED_COLS = ['b', 'h', 'bp', 'hp', 'fc', 'ALfyL', 'Atfyt/s']  # Ts optional
ALL_COLS = REQUIRED_COLS + ['Ts']

# =========================
# Header normalization
# =========================
CANON = ['b','h','bp','hp','fc','ALfyL','Atfyt/s','Ts']

def _norm_token(s: str) -> str:
    s = str(s)
    s = s.strip()
    s = s.replace('\u00b7', '')  # middle dot
    s = s.replace('·','').replace('×','x').replace('*','')
    s = s.replace("'", "p")  # prime -> p (b' -> bp)
    s = s.lower()
    s = re.sub(r'\(.*?\)', '', s)               # remove units in parentheses
    s = re.sub(r'[^a-z0-9/]+', '', s)           # drop spaces & punctuation
    return s

def _canonical_name(raw: str) -> Optional[str]:
    t = _norm_token(raw)

    direct = {
        'b':'b', 'h':'h', 'bp':'bp', 'hp':'hp', 'fc':'fc', 'ts':'Ts',
        'alfyl':'ALfyL', 'al_fyl':'ALfyL', 'alf y l'.replace(' ',''):'ALfyL',
        'atfyt/s':'Atfyt/s', 'atfyt_s':'Atfyt/s', 'atfyts':'Atfyt/s',
        'atfytovers':'Atfyt/s', 'atfytpers':'Atfyt/s'
    }
    if t in direct:
        return direct[t]

    # Heuristics
    if t in ('bmm','width','breadth','bsection'): return 'b'
    if t in ('hmm','height','depth','overallh','sectionh'): return 'h'
    if t in ('bprime','bpmm','b_p','bp_','bpm'): return 'bp'
    if t in ('hprime','hpmm','h_p','hp_','hpm'): return 'hp'
    if t in ('fcc','fck','fcm','fcomp','compressivestrength','fpc','f_c','fckmpa'): return 'fc'
    if 'alfyl' in t or ('al' in t and 'fyl' in t): return 'ALfyL'
    if 'atfyt' in t and ('/s' in t or 'pers' in t or t.endswith('s')): return 'Atfyt/s'
    if t.startswith('t') and 'knm' in t: return 'Ts'
    return None

def normalize_dataframe_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Try to coerce headers to canonical names; returns (renamed_df, mapping)."""
    mapping: Dict[str, str] = {}
    for c in list(df.columns):
        canon = _canonical_name(str(c))
        if canon is not None:
            mapping[c] = canon
    if not mapping:
        return df, mapping
    df2 = df.rename(columns=mapping)
    return df2, mapping

# =========================
# Derived features
# =========================
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['A0'] = 0.85 * df['bp'] * df['hp']
    df['ph'] = 2.0 * (df['bp'] + df['hp'])
    df['sqrt_fc'] = np.sqrt(df['fc'])
    return df

def check_columns(df: pd.DataFrame, need_target: bool=False):
    need = ALL_COLS if need_target else REQUIRED_COLS
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

# =========================
# Scaling helpers
# =========================
def build_scalers(train_df: pd.DataFrame):
    scalers = {
        'stirrup': StandardScaler().fit(train_df[FEATURE_GROUPS['stirrup']]),
        'longitudinal': StandardScaler().fit(train_df[FEATURE_GROUPS['longitudinal']]),
        'concrete': StandardScaler().fit(train_df[FEATURE_GROUPS['concrete']]),
        'target': StandardScaler().fit(train_df[['Ts']])
    }
    return scalers

def apply_scalers(df: pd.DataFrame, scalers):
    out = {
        'stirrup': scalers['stirrup'].transform(df[FEATURE_GROUPS['stirrup']]).astype(np.float32),
        'longitudinal': scalers['longitudinal'].transform(df[FEATURE_GROUPS['longitudinal']]).astype(np.float32),
        'concrete': scalers['concrete'].transform(df[FEATURE_GROUPS['concrete']]).astype(np.float32),
        'original': df.copy().astype(np.float32)
    }
    if 'target' in scalers and 'Ts' in df.columns:
        out['target'] = scalers['target'].transform(df[['Ts']]).astype(np.float32)
    return out

def hetero_from_row(dataset_dict, idx, include_target=True):
    data = HeteroData()
    for node_type in ['stirrup','longitudinal','concrete']:
        x = dataset_dict[node_type][idx].reshape(1,-1)
        data[node_type].x = torch.tensor(x, dtype=torch.float32)

        orig = dataset_dict['original'].iloc[idx][FEATURE_GROUPS[node_type]].values.reshape(1,-1)
        data[node_type].orig = torch.tensor(orig, dtype=torch.float32)
        data[node_type].edge_index = torch.tensor([[0],[0]], dtype=torch.long)

    data['stirrup','confines','longitudinal'].edge_index = torch.tensor([[0],[0]], dtype=torch.long)
    data['longitudinal','transfers','concrete'].edge_index = torch.tensor([[0],[0]], dtype=torch.long)
    data['stirrup','confines','concrete'].edge_index    = torch.tensor([[0],[0]], dtype=torch.long)

    if include_target and 'target' in dataset_dict:
        data.y = torch.tensor(dataset_dict['target'][idx], dtype=torch.float32)
    return data

class RowDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dict, include_target=True):
        self.dataset_dict = dataset_dict
        self.n = len(dataset_dict['original'])
        self.include_target = include_target
    def __len__(self): return self.n
    def __getitem__(self, idx):
        d = hetero_from_row(self.dataset_dict, idx, include_target=self.include_target)
        d.idx = idx
        return d

# =========================
# Model
# =========================
class EnhancedGNNLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.msg_s2l = MLP([2*hidden_dim, 2*hidden_dim, hidden_dim], norm="batch_norm")
        self.msg_l2c = MLP([2*hidden_dim, 2*hidden_dim, hidden_dim], norm="batch_norm")
        self.msg_s2c = MLP([2*hidden_dim, 2*hidden_dim, hidden_dim], norm="batch_norm")
        self.gru_s = GRUCell(hidden_dim, hidden_dim)
        self.gru_l = GRUCell(hidden_dim, hidden_dim)
        self.mlp_upd_c = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim))
        self.norm_s = LayerNorm(hidden_dim)
        self.norm_l = LayerNorm(hidden_dim)
        self.norm_c = LayerNorm(hidden_dim)

    def forward(self, x_dict, edge_index_dict):
        h_s, h_l, h_c = x_dict['stirrup'], x_dict['longitudinal'], x_dict['concrete']
        a_s = torch.zeros_like(h_s)
        a_l = torch.zeros_like(h_l)
        a_c = torch.zeros_like(h_c)

        key = ('stirrup','confines','longitudinal')
        if key in edge_index_dict:
            ei = edge_index_dict[key]
            m = self.msg_s2l(torch.cat([h_s[ei[0]], h_l[ei[1]]], dim=-1))
            a_l = a_l.index_add(0, ei[1], m)

        key = ('longitudinal','transfers','concrete')
        if key in edge_index_dict:
            ei = edge_index_dict[key]
            m = self.msg_l2c(torch.cat([h_l[ei[0]], h_c[ei[1]]], dim=-1))
            a_c = a_c.index_add(0, ei[1], m)

        key = ('stirrup','confines','concrete')
        if key in edge_index_dict:
            ei = edge_index_dict[key]
            m = self.msg_s2c(torch.cat([h_s[ei[0]], h_c[ei[1]]], dim=-1))
            a_c = a_c.index_add(0, ei[1], m)

        h_s_out = self.norm_s(h_s + self.gru_s(a_s, h_s))
        h_l_out = self.norm_l(h_l + self.gru_l(a_l, h_l))
        h_c_out = self.norm_c(h_c + self.mlp_upd_c(torch.cat([h_c, a_c], dim=-1)))
        return {'stirrup': h_s_out, 'longitudinal': h_l_out, 'concrete': h_c_out}

class PhysicsGNN(nn.Module):
    def __init__(self, hidden_dim=512, num_layers=6):
        super().__init__()
        self.encoders = nn.ModuleDict({
            'stirrup': nn.Sequential(MLP([5, hidden_dim, hidden_dim], norm="batch_norm"), nn.ReLU(), nn.Dropout(0.3)),
            'longitudinal': nn.Sequential(MLP([1, hidden_dim, hidden_dim], norm="batch_norm"), nn.ReLU(), nn.Dropout(0.3)),
            'concrete': nn.Sequential(MLP([4, hidden_dim, hidden_dim], norm="batch_norm"), nn.ReLU(), nn.Dropout(0.3)),
        })
        self.layers = nn.ModuleList([EnhancedGNNLayer(hidden_dim) for _ in range(num_layers)])
        self.t_s   = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 1))
        self.t_l   = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 1))
        self.t_c   = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 1))
        self.t_int = nn.Sequential(nn.Linear(3*hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
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
        h = {
            'stirrup':      self.encoders['stirrup'](data['stirrup'].x),
            'longitudinal': self.encoders['longitudinal'](data['longitudinal'].x),
            'concrete':     self.encoders['concrete'](data['concrete'].x),
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

# =========================
# Physics anchor loss
# =========================
def physics_anchor_loss(pred, batch, mean, scale):
    fc = batch['concrete'].orig[:, 0].float()
    bp = batch['stirrup'].orig[:, 1].float()
    hp = batch['stirrup'].orig[:, 2].float()
    Atfyt_s = batch['stirrup'].orig[:, 0].float()
    ALfyL = batch['longitudinal'].orig[:, 0].float()
    A0 = 0.85 * bp * hp
    ph = 2 * (bp + hp)
    T_conc = (0.33 * torch.sqrt(fc) * (A0**2) / ph) / 1e6
    T_st   = (2 * A0 * Atfyt_s) / 1e6
    T_long = (2 * A0 * ALfyL / ph) / 1e6
    T_phys = T_conc + T_st + T_long
    T_phys_norm = (T_phys - mean) / scale
    return F.huber_loss(pred, T_phys_norm, delta=0.5)

# =========================
# Eval helper
# =========================
@torch.no_grad()
def eval_phase(model, loader, device, target_scaler):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        batch = batch.to(device)
        pred_std = model(batch)
        pred = pred_std.cpu().numpy() * target_scaler.scale_ + target_scaler.mean_
        if hasattr(batch, 'y'):
            true = batch.y.cpu().numpy() * target_scaler.scale_ + target_scaler.mean_
        else:
            true = None
        preds.append(pred.reshape(-1))
        if true is not None:
            trues.append(true.reshape(-1))
    preds = np.concatenate(preds)
    trues = np.concatenate(trues) if trues else None
    out = {'preds': preds, 'trues': trues}
    if trues is not None:
        mse = np.mean((preds - trues)**2); rmse = np.sqrt(mse)
        mae = np.mean(np.abs(preds - trues)); r2 = r2_score(trues, preds)
        out.update({'rmse': rmse, 'mae': mae, 'r2': r2})
    return out

# =========================
# Training
# =========================
def train_model(df, hidden_dim=512, num_layers=6, batch_size=32, epochs=150, lr=1e-4,
                weight_decay=1e-5, lambda_int=1e-4, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_val, test = train_test_split(df, test_size=0.2, random_state=42)
    train, val = train_test_split(train_val, test_size=0.125, random_state=42)

    scalers = build_scalers(train)
    train_d = apply_scalers(train, scalers)
    val_d   = apply_scalers(val, scalers)
    test_d  = apply_scalers(test, scalers)

    tr_ds  = RowDataset(train_d, include_target=True)
    va_ds  = RowDataset(val_d, include_target=True)
    te_ds  = RowDataset(test_d, include_target=True)
    loaders = {
        'train': DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                            follow_batch=['stirrup','longitudinal','concrete']),
        'val':   DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                            follow_batch=['stirrup','longitudinal','concrete']),
        'test':  DataLoader(te_ds, batch_size=batch_size, shuffle=False,
                            follow_batch=['stirrup','longitudinal','concrete']),
    }

    model = PhysicsGNN(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10, verbose=False)

    scale = torch.tensor(scalers['target'].scale_.astype(np.float32), device=device)
    mean  = torch.tensor(scalers['target'].mean_.astype(np.float32),  device=device)

    hist = {'train': [], 'val': [], 'test': [], 'train_rmse': [], 'val_rmse': [], 'test_rmse': []}
    best_state, best_val = None, 1e9
    patience, no_improve = 25, 0

    progress = st.progress(0.0, text="Training...")
    status = st.empty()

    for epoch in range(1, epochs+1):
        model.train()
        total = 0.0
        physics_weight = max(0.1, 1.5 - epoch/100.0)
        for batch in loaders['train']:
            opt.zero_grad()
            batch = batch.to(device)
            pred = model(batch)
            loss_data = F.huber_loss(pred, batch.y, delta=0.5)
            loss_phys = physics_anchor_loss(pred, batch, mean, scale)
            loss_eps  = (model.epsilon()**2) * lambda_int
            loss = loss_data + physics_weight*loss_phys + loss_eps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        tr_loss = total / max(1, len(loaders['train']))
        hist['train'].append(tr_loss)

        val_out  = eval_phase(model, loaders['val'], device, scalers['target'])
        test_out = eval_phase(model, loaders['test'], device, scalers['target'])

        tr_out = eval_phase(model, loaders['train'], device, scalers['target'])
        hist['train_rmse'].append(tr_out['rmse'])
        hist['val_rmse'].append(val_out['rmse'])
        hist['test_rmse'].append(test_out['rmse'])

        sched.step(val_out['rmse'])
        alpha, beta, gamma, eps = model.nonneg_weights()
        status.write(
            f"Epoch {epoch}/{epochs} | Train Loss {tr_loss:.4f} | "
            f"Val RMSE {val_out['rmse']:.2f} | Test RMSE {test_out['rmse']:.2f} | "
            f"α:{alpha.item():.3f} β:{beta.item():.3f} γ:{gamma.item():.3f} ε:{eps.item():.5f}"
        )
        progress.progress(epoch/epochs)

        if val_out['rmse'] < best_val:
            best_val = val_out['rmse']; best_state = model.state_dict(); no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                st.info(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final = {
        'train': eval_phase(model, loaders['train'], device, scalers['target']),
        'val':   eval_phase(model, loaders['val'], device, scalers['target']),
        'test':  eval_phase(model, loaders['test'], device, scalers['target']),
        'scalers': scalers,
        'model': model,
        'history': hist,
        'splits': {'train': train, 'val': val, 'test': test}
    }
    return final

# =========================
# Sidebar settings
# =========================
with st.sidebar:
    st.header("Settings")
    hidden_dim   = st.select_slider("Hidden dim", options=[128,256,384,512,640], value=512)
    num_layers   = st.slider("GNN layers", 2, 8, 6, 1)
    batch_size   = st.slider("Batch size", 8, 128, 32, 8)
    epochs       = st.slider("Epochs", 20, 500, 150, 10)
    lr           = st.select_slider("Learning rate", options=[5e-5,1e-4,2e-4,5e-4,1e-3], value=1e-4)
    weight_decay = st.select_slider("Weight decay", options=[0.0,1e-6,1e-5,1e-4], value=1e-5)
    lambda_int   = st.select_slider("λ (interaction penalty)", options=[0.0,1e-5,1e-4,5e-4,1e-3], value=1e-4)

# =========================
# Tabs
# =========================
tab_data, tab_train, tab_infer, tab_viz = st.tabs(["📥 Data", "🧠 Train / Load", "🔮 Inference", "📊 Visualize"])

# =========================
# TAB: Data
# =========================
with tab_data:
    st.subheader("Load dataset (CSV)")
    st.caption("Accepts header variants (e.g., b', h', Atfyt_s, units like b (mm)) or headerless CSVs.")
    up = st.file_uploader("Upload CSV", type=['csv'])
    assume_no_header = st.checkbox("My CSV has no header row", value=False)
    df = None

    # Auto-load your Desktop CSV if present
    default_csv_path = r"C:\Users\vikas\Desktop\mix data.csv"
    if up is None and os.path.exists(default_csv_path):
        st.info(f"Auto-detected: {default_csv_path}")
        up = open(default_csv_path, "rb")

    if up is not None:
        try:
            if assume_no_header:
                tmp = pd.read_csv(up, header=None, sep=None, engine='python')
                if tmp.shape[1] < 7:
                    raise ValueError(f"Found {tmp.shape[1]} columns; expected at least 7.")
                cols = ['b','h','bp','hp','fc','ALfyL','Atfyt/s']
                if tmp.shape[1] >= 8:
                    cols.append('Ts')
                tmp = tmp.iloc[:, :len(cols)]
                tmp.columns = cols
                df = tmp
            else:
                up.seek(0)
                tmp = pd.read_csv(up, sep=None, engine='python')
                tmp_norm, mapping = normalize_dataframe_columns(tmp)
                needed = set(REQUIRED_COLS)
                if not (set(tmp_norm.columns) & needed):
                    up.seek(0)
                    tmp2 = pd.read_csv(up, header=None, sep=None, engine='python')
                    cols = ['b','h','bp','hp','fc','ALfyL','Atfyt/s']
                    if tmp2.shape[1] >= 8:
                        cols.append('Ts')
                    tmp2 = tmp2.iloc[:, :len(cols)]
                    tmp2.columns = cols
                    df = tmp2
                    st.info("Loaded as headerless CSV (assigned canonical column names).")
                else:
                    df = tmp_norm
                    if mapping:
                        st.info(f"Renamed columns: {mapping}")
            need_target = 'Ts' in df.columns
            check_columns(df, need_target=need_target)
            df = add_derived_features(df)
            st.success(f"Loaded {len(df)} rows. Derived features added.")
            st.dataframe(df.head())
            st.session_state['df_loaded'] = df
        except Exception as e:
            st.error(f"Load failed: {e}")

# =========================
# TAB: Train / Load
# =========================
with tab_train:
    st.subheader("Train model or load checkpoint")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Train now", type="primary", disabled=('df_loaded' not in st.session_state)):
            with st.spinner("Training..."):
                df_train = st.session_state['df_loaded']
                if 'Ts' not in df_train.columns:
                    st.error("Training requires Ts column.")
                else:
                    results = train_model(
                        df_train, hidden_dim=hidden_dim, num_layers=num_layers,
                        batch_size=batch_size, epochs=epochs, lr=lr,
                        weight_decay=weight_decay, lambda_int=lambda_int
                    )
                    st.session_state['model']   = results['model']
                    st.session_state['scalers'] = results['scalers']
                    st.session_state['history'] = results['history']
                    st.session_state['splits']  = results['splits']
                    st.session_state['eval']    = results
                    # Save locally next to script
                    torch.save(results['model'].state_dict(), "final_model.pth")
                    joblib.dump(results['scalers'], "scalers.joblib")
                    st.success("Training complete. Saved final_model.pth and scalers.joblib")

    with col2:
        st.markdown("**Load existing checkpoint**")
        ckpt_file = st.file_uploader("Load model (.pth)", type=['pth'], key="ckpt")
        scal_file = st.file_uploader("Load scalers (joblib)", type=['joblib'], key="scal")
        if st.button("Load model + scalers"):
            try:
                model = PhysicsGNN(hidden_dim=hidden_dim, num_layers=num_layers)
                if ckpt_file is not None:
                    bytes_io = io.BytesIO(ckpt_file.read())
                    state = torch.load(bytes_io, map_location='cpu')
                else:
                    state = torch.load("final_model.pth", map_location='cpu')
                model.load_state_dict(state)
                if scal_file is not None:
                    scalers = joblib.load(scal_file)
                else:
                    scalers = joblib.load("scalers.joblib")
                st.session_state['model'] = model
                st.session_state['scalers'] = scalers
                st.success("Loaded model & scalers.")
            except Exception as e:
                st.error(f"Load failed: {e}")

# =========================
# TAB: Inference
# =========================
with tab_infer:
    st.subheader("Predict torsional strength")
    if ('model' not in st.session_state) or ('scalers' not in st.session_state):
        st.info("Train or load a model first (see previous tab).")
    else:
        model   = st.session_state['model'].eval()
        scalers = st.session_state['scalers']
        device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        st.markdown("**Single-specimen form**")
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            b  = st.number_input("b (mm)", value=350.0)
            h  = st.number_input("h (mm)", value=500.0)
        with c2:
            bp = st.number_input("b' (mm)", value=300.0)
            hp = st.number_input("h' (mm)", value=450.0)
        with c3:
            fc = st.number_input("fc (MPa)", value=40.0)
            AL = st.number_input("AL·fyL (kN)", value=300.0)
        with c4:
            At = st.number_input("At·fyt/s (N/mm)", value=300.0)

        if st.button("Predict (single)"):
            row = pd.DataFrame([{'b':b,'h':h,'bp':bp,'hp':hp,'fc':fc,'ALfyL':AL,'Atfyt/s':At}])
            row = add_derived_features(row)
            dct = apply_scalers(row, scalers)
            data = hetero_from_row(dct, 0, include_target=False).to(device)
            with torch.no_grad():
                pred_std = model(data)
                pred = (pred_std.cpu().numpy().reshape(-1) * scalers['target'].scale_ + scalers['target'].mean_)[0]
            st.success(f"Predicted torsional strength: **{pred:.2f} kN·m**")

        st.divider()
        st.markdown("**Batch inference (CSV)** — columns: " + ", ".join(REQUIRED_COLS))
        inf_up = st.file_uploader("Upload CSV for batch prediction", type=['csv'], key="infercsv")
        if inf_up is not None:
            try:
                df_inf = pd.read_csv(inf_up, sep=None, engine='python')
                # Try normalize; if nothing recognized, fallback to headerless attempt
                df_inf2, map2 = normalize_dataframe_columns(df_inf)
                if not set(REQUIRED_COLS).issubset(set(df_inf2.columns)):
                    inf_up.seek(0)
                    df_inf = pd.read_csv(inf_up, header=None, sep=None, engine='python')
                    cols = ['b','h','bp','hp','fc','ALfyL','Atfyt/s']
                    df_inf = df_inf.iloc[:, :len(cols)]
                    df_inf.columns = cols
                    df_inf2 = df_inf
                check_columns(df_inf2, need_target=False)
                df_inf2 = add_derived_features(df_inf2)
                dct = apply_scalers(df_inf2, scalers)
                ds  = RowDataset(dct, include_target=False)
                loader = DataLoader(ds, batch_size=64, shuffle=False, follow_batch=['stirrup','longitudinal','concrete'])
                preds = []
                with torch.no_grad():
                    for batch in loader:
                        batch = batch.to(device)
                        p = model(batch).cpu().numpy().reshape(-1)
                        p = p * scalers['target'].scale_ + scalers['target'].mean_
                        preds.append(p)
                preds = np.concatenate(preds)
                out_df = df_inf2[REQUIRED_COLS].copy()
                out_df['Ts_pred'] = preds
                st.success(f"Predicted {len(out_df)} rows.")
                st.dataframe(out_df.head())
                csv = out_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download predictions CSV", data=csv, file_name="torsion_predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Batch inference failed: {e}")

# =========================
# TAB: Visualize
# =========================
with tab_viz:
    st.subheader("Training curves & diagnostics")
    if 'history' not in st.session_state or 'eval' not in st.session_state:
        st.info("Train a model to see learning curves.")
    else:
        hist = st.session_state['history']
        eval_all = st.session_state['eval']
        splits = st.session_state['splits']

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(hist['train_rmse'], label='Train RMSE')
            ax.plot(hist['val_rmse'], label='Val RMSE')
            ax.plot(hist['test_rmse'], label='Test RMSE')
            ax.set_xlabel('Epoch'); ax.set_ylabel('RMSE (kN·m)'); ax.set_title('RMSE vs Epoch')
            ax.grid(True, ls='--', alpha=0.6); ax.legend()
            st.pyplot(fig)

        with c2:
            tr = eval_all['train']; va = eval_all['val']; te = eval_all['test']
            fig, ax = plt.subplots(figsize=(6,6))
            for label, out, color, marker in [
                ('Train', tr, 'red', 'o'),
                ('Val',   va, 'green', 's'),
                ('Test',  te, 'blue', '^')
            ]:
                ax.scatter(out['trues'], out['preds'], c=color, alpha=0.65, edgecolors='k', s=60,
                           label=f"{label} (R²={out['r2']:.3f})")
            all_t = np.concatenate([tr['trues'], va['trues'], te['trues']])
            all_p = np.concatenate([tr['preds'], va['preds']])
            all_p = np.concatenate([all_p, te['preds']])
            mn, mx = float(min(all_t.min(), all_p.min())), float(max(all_t.max(), all_p.max()))
            ax.plot([mn, mx], [mn, mx], 'k--', lw=2)
            ax.set_xlabel('Actual (kN·m)'); ax.set_ylabel('Predicted (kN·m)')
            ax.set_title('Pred vs Actual (All Phases)')
            ax.grid(True, ls='--', alpha=0.6); ax.legend()
            st.pyplot(fig)

        st.markdown("**Residuals (Test)**")
        te = eval_all['test']
        resid = te['trues'] - te['preds']
        c3, c4 = st.columns(2)
        with c3:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.scatterplot(x=te['preds'], y=resid, ax=ax, alpha=0.7, edgecolor='k')
            ax.axhline(0, color='r', ls='--', lw=2)
            ax.set_xlabel('Predicted (kN·m)'); ax.set_ylabel('Residual (kN·m)')
            ax.set_title('Residuals vs Predicted (Test)')
            ax.grid(True, ls='--', alpha=0.6)
            st.pyplot(fig)
        with c4:
            from scipy import stats
            fig, ax = plt.subplots(figsize=(6,4))
            stats.probplot(resid, dist="norm", plot=ax)
            ax.get_lines()[0].set_markerfacecolor('C0'); ax.get_lines()[0].set_markeredgecolor('k')
            ax.get_lines()[1].set_color('r'); ax.get_lines()[1].set_linewidth(2.5)
            ax.set_title('QQ Plot (Test Residuals)')
            st.pyplot(fig)

        if st.button("Export combined predictions (Train/Val/Test)"):
            train_df = splits['train'].copy()
            val_df   = splits['val'].copy()
            test_df  = splits['test'].copy()
            train_df['phase'] = 'train'; val_df['phase'] = 'val'; test_df['phase'] = 'test'
            train_df = train_df.reset_index(drop=True)
            val_df   = val_df.reset_index(drop=True)
            test_df  = test_df.reset_index(drop=True)

            def add_preds(df_in):
                dfa = add_derived_features(df_in)
                dct = apply_scalers(dfa, st.session_state['scalers'])
                ds = RowDataset(dct, include_target=('Ts' in df_in.columns))
                loader = DataLoader(ds, batch_size=64, shuffle=False,
                                    follow_batch=['stirrup','longitudinal','concrete'])
                preds = []
                model = st.session_state['model'].eval()
                device = next(model.parameters()).device
                with torch.no_grad():
                    for batch in loader:
                        batch = batch.to(device)
                        p = model(batch).cpu().numpy().reshape(-1)
                        p = p * st.session_state['scalers']['target'].scale_ + st.session_state['scalers']['target'].mean_
                        preds.append(p)
                preds = np.concatenate(preds)
                out = df_in.copy()
                out['Ts_pred'] = preds
                if 'Ts' in out.columns:
                    out['residual'] = out['Ts'] - out['Ts_pred']
                return out

            out_df = pd.concat([add_preds(train_df), add_preds(val_df), add_preds(test_df)], ignore_index=True)
            csv = out_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download combined_predictions.csv", data=csv,
                               file_name="combined_predictions.csv", mime="text/csv")
