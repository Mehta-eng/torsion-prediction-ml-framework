import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from bayes_opt import BayesianOptimization
import joblib

# ======================
# Global Matplotlib Style
# ======================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 10
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# ======================
# 1. Configuration
# ======================
if len(sys.argv) > 1:
    DATA_PATH = sys.argv[1]
else:
    DATA_PATH = r'C:\Users\vikas\Desktop\Synthetic data\torsional\mix data.csv'  # Default path

# Choose code for torsion cap: 'ACI', 'EC2', or 'PHYS' (fallback)
CODE_CAP = os.environ.get('CODE_CAP', 'ACI').upper()

# Code parameters (tune to your member details)
CODE_PARAMS = {
    'ACI': {
        'phi': 0.75,         # strength reduction factor
        'theta_deg': 40.0,   # strut angle to member axis
        'ao_over_A0': 0.85   # Ao_t / A0 ratio
    },
    'EC2': {
        'nu1': 0.6,          # concrete factor (≈0.6*(1-fck/250))
        'theta_deg': 45.0,   # truss angle
        'gamma_c': 1.0,      # partial factor for concrete
        'ao_over_A0': 0.85
    }
}

TARGET = 'Ts'
COLORS = {
    'train': '#0D47A1',   # Darker Blue
    'val': '#1B5E20',     # Dark Green
    'test': '#E65100',    # Deep Orange
    'ensemble': '#6A1B9A',  # Deep Purple
    'pgnn': '#0D47A1',    # Darker Blue
    'gnn': '#E65100',     # Deep Orange
    'xgb': '#B71C1C',     # Dark Red
    'bo_stack': '#4E342E'   # Dark Brown
}

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ======================
# 2. Utilities
# ======================
def evaluate_model(y_true, y_pred, model_name, set_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    bias = np.mean(y_true - y_pred)

    print(f"{model_name} ({set_name}):")
    print(f"  RMSE = {rmse:.2f} kNm")
    print(f"  MAE = {mae:.2f} kNm")
    print(f"  R² = {r2:.4f}")
    print(f"  Bias = {bias:.2f} kNm")
    print("-" * 40)
    return rmse, mae, r2, bias


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

# ======================
# 3. Data Preparation (with validation split)
# ======================

# ---- Code-based torsion caps ----
# All returns are in kNm. Uses row with fields: b, h, ph, A0, Atfyt/s, ALfyl, T_concrete, T_stirrup, T_long.

def aci_torsion_cap(row, params):
    """Approximate ACI 318 cap using thin-walled truss model.
    Tn ≈ 2 * (Ao_t) * (At fyt / s) * (Ao_t / p) * cot(theta).
    Then φTn is the design cap. Units: Nmm converted to kNm.
    Notes:
      - Ao_t = ao_over_A0 * A0, with A0 already ≈0.85 b h in data.
      - p is perimeter 'ph' (mm).
      - This is a practical approximation; set theta, ao_over_A0, phi per your member.
    """
    phi = CODE_PARAMS['ACI'].get('phi', 0.75)
    theta = np.deg2rad(CODE_PARAMS['ACI'].get('theta_deg', 40.0))
    ao_over_A0 = CODE_PARAMS['ACI'].get('ao_over_A0', 0.85)
    Ao_t = ao_over_A0 * row['A0']  # mm^2
    q_st = (row['Atfyt/s'])  # N/mm
    Tn_Nmm = 2.0 * Ao_t * q_st * (Ao_t / max(row['ph'], 1e-9)) * (1.0 / np.tan(theta))
    Tn_kNm = Tn_Nmm * 1e-6
    return phi * Tn_kNm


def ec2_torsion_cap(row, params):
    """Approximate EC2 cap via truss model with concrete factor ν1.
    T_Rd ≈ ν1 * (Ao_t / p) * (2 * Ao_t * (At fyt / s)) * cot(theta) / γ_c.
    """
    nu1 = CODE_PARAMS['EC2'].get('nu1', 0.6)
    gamma_c = CODE_PARAMS['EC2'].get('gamma_c', 1.0)
    theta = np.deg2rad(CODE_PARAMS['EC2'].get('theta_deg', 45.0))
    ao_over_A0 = CODE_PARAMS['EC2'].get('ao_over_A0', 0.85)
    Ao_t = ao_over_A0 * row['A0']
    q_st = (row['Atfyt/s'])
    Tn_Nmm = nu1 / max(gamma_c, 1e-9) * (Ao_t / max(row['ph'], 1e-9)) * (2.0 * Ao_t * q_st) * (1.0 / np.tan(theta))
    Tn_kNm = Tn_Nmm * 1e-6
    return Tn_kNm


def physics_sum_cap(row):
    return 1.10 * (row['T_concrete'] + row['T_stirrup'] + row['T_long'])


def torsion_code_cap(row):
    if CODE_CAP == 'ACI':
        return aci_torsion_cap(row, CODE_PARAMS['ACI'])
    elif CODE_CAP == 'EC2':
        return ec2_torsion_cap(row, CODE_PARAMS['EC2'])
    else:
        return physics_sum_cap(row)


def load_and_preprocess_data():
    data = pd.read_csv(DATA_PATH)

    # Debug: original columns
    print("Original columns:", data.columns.tolist())

    # Clean column names
    data.columns = (data.columns.str.strip()
                    .str.replace(' ', '', regex=False)
                    .str.replace("'", '', regex=False)
                    .str.replace('"', '', regex=False))
    print("Cleaned columns:", data.columns.tolist())

    # Deduplicate columns
    new_columns = []
    seen = {}
    for col in data.columns:
        if col in seen:
            seen[col] += 1
            new_columns.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_columns.append(col)
    data.columns = new_columns
    print("Deduplicated columns:", data.columns.tolist())

    # Map longitudinal reinforcement column to 'ALfyl'
    longitudinal_cols = [c for c in data.columns if 'AL' in c and ('fy' in c or 'fyl' in c)]
    if not longitudinal_cols:
        raise KeyError("Could not find longitudinal reinforcement column in dataset")
    data.rename(columns={longitudinal_cols[0]: 'ALfyl'}, inplace=True)
    print(f"Using longitudinal column: {longitudinal_cols[0]} -> ALfyl")

    # Map stirrup column to 'Atfyt/s'
    stirrup_cols = [c for c in data.columns if ('Atfyt' in c) or ('Atfy' in c)]
    if not stirrup_cols:
        raise KeyError("Could not find stirrup reinforcement column in dataset")
    data.rename(columns={stirrup_cols[0]: 'Atfyt/s'}, inplace=True)
    print(f"Using stirrup column: {stirrup_cols[0]} -> Atfyt/s")

    # Base physics feature engineering (units assumed mm, MPa, N, etc.)
    data['A0'] = 0.85 * data['b'] * data['h']  # mm²
    data['ph'] = 2 * (data['b'] + data['h'])   # mm
    data['sqrt_fc'] = np.sqrt(data['fc'])      # sqrt(MPa)

    # Concrete torsion contribution (kNm)
    data['T_concrete'] = 0.33 * data['sqrt_fc'] * (data['A0'] ** 2) / data['ph'] * 1e-6
    # Stirrups torsion contribution (kNm)
    data['T_stirrup'] = 2 * data['A0'] * data['Atfyt/s'] * 1e-6
    # Longitudinal torsion contribution (kNm)
    data['T_long'] = (2 * data['A0'] * data['ALfyl']) / (data['ph'] * 1000.0)

    # Physics-based target (kNm)
    data['T_phy'] = data['T_concrete'] + data['T_stirrup'] + data['T_long']
    # Code cap (kNm)
    data['T_cap'] = data.apply(torsion_code_cap, axis=1)

    # Restrict features used by PGNN/XGB
    features = ['b', 'h', 'fc', 'ALfyl', 'Atfyt/s', 'A0', 'ph', 'sqrt_fc', 'T_concrete', 'T_stirrup', 'T_long']

    # Keep required columns
    needed = features + [TARGET, 'T_phy', 'T_cap']
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise KeyError(f"Missing required columns after preprocessing: {missing}")

    # Split while preserving alignment among X, y, T_phy, T_cap
    full = data[needed].copy()
    X = full[features]
    y = full[TARGET].values.astype(np.float32)
    T_phy = full['T_phy'].values.astype(np.float32)
    T_cap = full['T_cap'].values.astype(np.float32)

    X_train_val, X_test, y_train_val, y_test, Tphy_train_val, Tphy_test, Tcap_train_val, Tcap_test = \
        train_test_split(X, y, T_phy, T_cap, test_size=0.2, random_state=SEED)

    X_train, X_val, y_train, y_val, Tphy_train, Tphy_val, Tcap_train, Tcap_val = \
        train_test_split(X_train_val, y_train_val, Tphy_train_val, Tcap_train_val, test_size=0.125, random_state=SEED)

    # Normalize features only (targets and physics terms stay in kNm)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    # Build DataFrames for GNN node packing
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features)
    X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=features)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features)

    def create_graph_data(X_df):
        graph_data = []
        for _, row in X_df.iterrows():
            concrete = [row['b'], row['h'], row['fc'], row['sqrt_fc'], row['A0'], row['ph'], row['T_concrete']]
            stirrup = [row['Atfyt/s'], row['T_stirrup']]
            longitudinal = [row['ALfyl'], row['T_long']]
            graph_data.append([concrete, stirrup, longitudinal])
        return graph_data

    X_train_graph = create_graph_data(X_train_scaled_df)
    X_val_graph = create_graph_data(X_val_scaled_df)
    X_test_graph = create_graph_data(X_test_scaled_df)

    return {
        'features': features,
        'X_train': X_train_scaled.astype(np.float32),
        'X_val': X_val_scaled.astype(np.float32),
        'X_test': X_test_scaled.astype(np.float32),
        'X_train_graph': X_train_graph,
        'X_val_graph': X_val_graph,
        'X_test_graph': X_test_graph,
        'y_train': y_train.astype(np.float32),
        'y_val': y_val.astype(np.float32),
        'y_test': y_test.astype(np.float32),
        'T_phy_train': Tphy_train,
        'T_phy_val': Tphy_val,
        'T_phy_test': Tphy_test,
        'T_cap_train': Tcap_train,
        'T_cap_val': Tcap_val,
        'T_cap_test': Tcap_test,
        'scaler_X': scaler_X
    }

# ======================
# 4. Models
# ======================
class PhysicsGuidedNN(nn.Module):
    def __init__(self, input_size=11, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.apply(init_weights)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.output(x)


class PhysicsGuidedGNN(nn.Module):
    def __init__(self, concrete_dim=7, stirrup_dim=2, long_dim=2, hidden_dim=128, K=3):
        super().__init__()
        self.K = K
        # Encoders
        self.concrete_encoder = nn.Linear(concrete_dim, hidden_dim)
        self.stirrup_encoder  = nn.Linear(stirrup_dim, hidden_dim)
        self.long_encoder     = nn.Linear(long_dim, hidden_dim)
        # Shared message MLP
        self.msg_mlp = nn.Linear(2 * hidden_dim, hidden_dim)
        # Node update GRUs
        self.gru_s = nn.GRUCell(hidden_dim, hidden_dim)
        self.gru_l = nn.GRUCell(hidden_dim, hidden_dim)
        self.gru_c = nn.GRUCell(hidden_dim, hidden_dim)
        # Readout
        self.readout = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.apply(init_weights)

    def forward(self, nodes):
        concrete, stirrup, longitudinal = nodes
        h_c = F.relu(self.concrete_encoder(concrete))
        h_s = F.relu(self.stirrup_encoder(stirrup))
        h_l = F.relu(self.long_encoder(longitudinal))
        for _ in range(self.K):
            # Bidirectional messaging among nodes
            msg_c_to_s = F.relu(self.msg_mlp(torch.cat([h_c, h_s], dim=1)))
            msg_s_to_l = F.relu(self.msg_mlp(torch.cat([h_s, h_l], dim=1)))
            msg_l_to_c = F.relu(self.msg_mlp(torch.cat([h_l, h_c], dim=1)))
            # GRU updates per node
            h_s = self.gru_s(msg_c_to_s, h_s)
            h_l = self.gru_l(msg_s_to_l, h_l)
            h_c = self.gru_c(msg_l_to_c, h_c)
        combined = torch.cat([h_s, h_l, h_c], dim=1)
        return self.readout(combined).squeeze()


class PGNN_Trainer:
    def __init__(self, X_train, y_train, X_val, y_val,
                 T_phy_train, T_phy_val, T_cap_train, T_cap_val,
                 lam_data=1.0, lam_phy=0.3, lam_code=0.3):
        self.X_train = torch.FloatTensor(X_train)
        self.y_train = torch.FloatTensor(y_train)
        self.X_val = torch.FloatTensor(X_val)
        self.y_val = torch.FloatTensor(y_val)
        self.T_phy_train = torch.FloatTensor(T_phy_train)
        self.T_phy_val = torch.FloatTensor(T_phy_val)
        self.T_cap_train = torch.FloatTensor(T_cap_train)
        self.T_cap_val = torch.FloatTensor(T_cap_val)
        self.model = PhysicsGuidedNN(input_size=X_train.shape[1])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=15, verbose=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.train_losses = []
        self.val_losses = []
        self.best_val_rmse = float('inf')
        self.best_model = None
        self.lam_data, self.lam_phy, self.lam_code = lam_data, lam_phy, lam_code

    def _make_train_loader(self, batch_size=32):
        dataset = TensorDataset(self.X_train, self.y_train, self.T_phy_train, self.T_cap_train)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train(self, epochs=300, batch_size=32, patience=20):
        train_loader = self._make_train_loader(batch_size)
        no_improve = 0
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for Xb, yb, Tphy_b, Tcap_b in train_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                Tphy_b, Tcap_b = Tphy_b.to(self.device), Tcap_b.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(Xb).squeeze()
                data_loss = F.mse_loss(preds, yb)
                phys_loss = F.mse_loss(preds, Tphy_b)
                code_viols = torch.clamp(preds - Tcap_b, min=0.0)
                code_loss = torch.mean(code_viols ** 2)
                loss = self.lam_data * data_loss + self.lam_phy * phys_loss + self.lam_code * code_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / max(1, len(train_loader))
            self.train_losses.append(avg_train_loss)

            # Validation RMSE (on data term only)
            val_rmse = self.validate()
            self.val_losses.append(val_rmse)
            self.scheduler.step(val_rmse)

            if val_rmse < self.best_val_rmse:
                self.best_val_rmse = val_rmse
                self.best_model = self.model.state_dict()
                no_improve = 0
            else:
                no_improve += 1

            if (epoch + 1) % 10 == 0:
                print(f'PGNN Epoch {epoch+1:03d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val RMSE: {val_rmse:.2f}')
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        if self.best_model is not None:
            self.model.load_state_dict(self.best_model)
        torch.save(self.model.state_dict(), 'pgnn_model.pth')
        return self

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            inputs = self.X_val.to(self.device)
            outputs = self.model(inputs).cpu().numpy().flatten()
        return np.sqrt(mean_squared_error(self.y_val.numpy(), outputs))

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy().flatten()
        return predictions


class GraphDataset(Dataset):
    def __init__(self, X_graph, y):
        self.X_graph = X_graph
        self.y = y
    def __len__(self):
        return len(self.X_graph)
    def __getitem__(self, idx):
        concrete = torch.FloatTensor(self.X_graph[idx][0])
        stirrup = torch.FloatTensor(self.X_graph[idx][1])
        longitudinal = torch.FloatTensor(self.X_graph[idx][2])
        return (concrete, stirrup, longitudinal), self.y[idx]


class GNN_Trainer:
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        sample = X_train[0]
        self.concrete_dim = len(sample[0])
        self.stirrup_dim = len(sample[1])
        self.long_dim = len(sample[2])
        print(f"Node dimensions: Concrete={self.concrete_dim}, Stirrup={self.stirrup_dim}, Longitudinal={self.long_dim}")
        self.model = PhysicsGuidedGNN(concrete_dim=self.concrete_dim, stirrup_dim=self.stirrup_dim, long_dim=self.long_dim, hidden_dim=128, K=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=15, verbose=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.train_losses = []
        self.val_losses = []
        self.best_val_rmse = float('inf')
        self.best_model = None

    def collate_fn(self, batch):
        nodes_list, targets_list = zip(*batch)
        concrete_list, stirrup_list, long_list = zip(*nodes_list)
        concrete_batch = torch.stack(concrete_list)
        stirrup_batch = torch.stack(stirrup_list)
        long_batch = torch.stack(long_list)
        targets_batch = torch.tensor(targets_list, dtype=torch.float32)
        return (concrete_batch, stirrup_batch, long_batch), targets_batch

    def train(self, epochs=300, batch_size=32, patience=20):
        train_dataset = GraphDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        no_improve = 0
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch_nodes, batch_targets in train_loader:
                concrete, stirrup, longitudinal = batch_nodes
                concrete = concrete.to(self.device)
                stirrup = stirrup.to(self.device)
                longitudinal = longitudinal.to(self.device)
                batch_targets = batch_targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model((concrete, stirrup, longitudinal))
                loss = F.mse_loss(outputs, batch_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / max(1, len(train_loader))
            self.train_losses.append(avg_train_loss)
            val_rmse = self.validate()
            self.val_losses.append(val_rmse)
            self.scheduler.step(val_rmse)
            if val_rmse < self.best_val_rmse:
                self.best_val_rmse = val_rmse
                self.best_model = self.model.state_dict()
                no_improve = 0
            else:
                no_improve += 1
            if (epoch + 1) % 10 == 0:
                print(f'GNN Epoch {epoch+1:03d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val RMSE: {val_rmse:.2f}')
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        if self.best_model is not None:
            self.model.load_state_dict(self.best_model)
        torch.save(self.model.state_dict(), 'gnn_model.pth')
        return self

    def validate(self):
        self.model.eval()
        val_dataset = GraphDataset(self.X_val, self.y_val)
        val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=self.collate_fn)
        predictions = []
        with torch.no_grad():
            for batch_nodes, _ in val_loader:
                concrete, stirrup, longitudinal = batch_nodes
                concrete = concrete.to(self.device)
                stirrup = stirrup.to(self.device)
                longitudinal = longitudinal.to(self.device)
                outputs = self.model((concrete, stirrup, longitudinal))
                predictions.append(outputs.cpu())
        predictions = torch.cat(predictions).numpy()
        return np.sqrt(mean_squared_error(self.y_val, predictions))

    def predict(self, X):
        self.model.eval()
        test_dataset = GraphDataset(X, np.zeros(len(X), dtype=np.float32))
        test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=self.collate_fn)
        predictions = []
        with torch.no_grad():
            for batch_nodes, _ in test_loader:
                concrete, stirrup, longitudinal = batch_nodes
                concrete = concrete.to(self.device)
                stirrup = stirrup.to(self.device)
                longitudinal = longitudinal.to(self.device)
                outputs = self.model((concrete, stirrup, longitudinal))
                predictions.append(outputs.cpu())
        predictions = torch.cat(predictions).numpy()
        return predictions


class XGBoost_Trainer:
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'seed': SEED,
            'alpha': 0.1,
            'lambda': 1.0,
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        self.model = None

    def train(self):
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        dval = xgb.DMatrix(self.X_val, label=self.y_val)
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=50
        )
        self.model.save_model('xgb_model.json')
        return self

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

# ======================
# 5. Bayesian-Optimized Stacking
# ======================
class BO_Stack:
    def __init__(self, pgnn_val, gnn_val, xgb_val, y_val):
        self.pgnn_val = pgnn_val
        self.gnn_val = gnn_val
        self.xgb_val = xgb_val
        self.y_val = y_val
        self.pbounds = {'w1': (0.1, 0.9), 'w2': (0.1, 0.9), 'w3': (0.1, 0.9), 'bias': (-100, 100)}
        self.weights = None

    def ensemble_objective(self, w1, w2, w3, bias):
        # Explicit sum-to-1 constraint
        total = w1 + w2 + w3 + 1e-9
        w1, w2, w3 = w1/total, w2/total, w3/total
        pred = w1*self.pgnn_val + w2*self.gnn_val + w3*self.xgb_val + bias
        rmse = np.sqrt(mean_squared_error(self.y_val, pred))
        return -rmse  # pure RMSE minimization

    def optimize_weights(self, init_points=10, n_iter=30):
        optimizer = BayesianOptimization(f=self.ensemble_objective, pbounds=self.pbounds, random_state=SEED, verbose=2)
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
        top_params = sorted(optimizer.res, key=lambda x: x['target'], reverse=True)[:5]
        avg_params = {
            'w1': np.mean([r['params']['w1'] for r in top_params]),
            'w2': np.mean([r['params']['w2'] for r in top_params]),
            'w3': np.mean([r['params']['w3'] for r in top_params]),
            'bias': np.mean([r['params']['bias'] for r in top_params])
        }
        total = avg_params['w1'] + avg_params['w2'] + avg_params['w3'] + 1e-9
        self.weights = {
            'w1': avg_params['w1']/total,
            'w2': avg_params['w2']/total,
            'w3': avg_params['w3']/total,
            'bias': avg_params['bias']
        }
        return self.weights

    def predict(self, pgnn_pred, gnn_pred, xgb_pred):
        # Static BO weights (no dynamic blending)
        return (self.weights['w1'] * pgnn_pred +
                self.weights['w2'] * gnn_pred +
                self.weights['w3'] * xgb_pred +
                self.weights['bias'])

# ======================
# 6. Visualization
# ======================
def plot_bo_residuals(y_true_train, y_pred_train, y_true_val, y_pred_val, y_true_test, y_pred_test):
    fig = plt.figure(figsize=(8, 6), dpi=150)
    residuals_train = y_true_train - y_pred_train
    residuals_val = y_true_val - y_pred_val
    residuals_test = y_true_test - y_pred_test
    plt.scatter(y_pred_train, residuals_train, c=COLORS['train'], alpha=0.7, s=100, edgecolor=COLORS['train'], linewidths=0.5, marker='o', label='Training Set')
    plt.scatter(y_pred_val, residuals_val, c=COLORS['val'], alpha=0.7, s=100, edgecolor=COLORS['val'], linewidths=0.5, marker='s', label='Validation Set')
    plt.scatter(y_pred_test, residuals_test, c=COLORS['test'], alpha=0.7, s=100, edgecolor=COLORS['test'], linewidths=0.5, marker='^', label='Testing Set')
    plt.axhline(0, color='k', linestyle='--', lw=1)
    plt.xlabel('Predicted Ts (kNm)')
    plt.ylabel('Residuals (kNm)')
    plt.title('BO-Stack Residual Analysis')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    stats_text = (f'Training Residuals:\nMean = {np.mean(residuals_train):.2f} kNm\nStd = {np.std(residuals_train):.2f} kNm\n\n'
                  f'Validation Residuals:\nMean = {np.mean(residuals_val):.2f} kNm\nStd = {np.std(residuals_val):.2f} kNm\n\n'
                  f'Test Residuals:\nMean = {np.mean(residuals_test):.2f} kNm\nStd = {np.std(residuals_test):.2f} kNm')
    plt.gca().annotate(stats_text, xy=(0.95, 0.05), xycoords='axes fraction', fontsize=9, va='bottom', ha='right', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5, alpha=0.9))
    plt.tight_layout(pad=1.5)
    plt.savefig('residual_analysis.jpg', bbox_inches='tight')
    plt.show()


def plot_combined_training_history(pgnn_trainer, gnn_trainer):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    ax1.plot(pgnn_trainer.train_losses, label='PGNN Training Loss', color=COLORS['pgnn'], linewidth=2, linestyle='-')
    ax1.plot(gnn_trainer.train_losses, label='GNN Training Loss', color=COLORS['gnn'], linewidth=2, linestyle='-')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss (composite / MSE)')
    ax1.set_title('Training Loss Comparison')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    ax2.plot(pgnn_trainer.val_losses, label='PGNN Validation RMSE', color=COLORS['pgnn'], linewidth=2, linestyle='-')
    ax2.plot(gnn_trainer.val_losses, label='GNN Validation RMSE', color=COLORS['gnn'], linewidth=2, linestyle='-')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Validation RMSE (kNm)')
    ax2.set_title('Validation Performance')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    plt.tight_layout(pad=2.0)
    plt.savefig('combined_training_history.jpg', bbox_inches='tight')
    plt.show()


def plot_model_comparison(y_true_train, y_preds_train, y_true_val, y_preds_val, y_true_test, y_preds_test, model_names):
    fig = plt.figure(figsize=(12, 9), dpi=150)
    model_colors = [COLORS['pgnn'], COLORS['gnn'], COLORS['xgb'], COLORS['bo_stack']]
    markers = ['o', 's', '^']
    marker_size = 80
    all_values = np.concatenate([y_true_train, y_true_val, y_true_test] + [p for p in y_preds_train] + [p for p in y_preds_val] + [p for p in y_preds_test])
    max_val = np.max(all_values)
    min_limit = 0.0
    max_limit = max_val * 1.05
    plt.plot([min_limit, max_limit], [min_limit, max_limit], 'k--', lw=2.0, label='Perfect Prediction')
    for i, (name, pred) in enumerate(zip(model_names, y_preds_train)):
        plt.scatter(y_true_train, pred, c=model_colors[i], alpha=0.7, s=marker_size, edgecolor=model_colors[i], linewidths=0.8, marker=markers[0], label=f'{name} (Train)')
    for i, (name, pred) in enumerate(zip(model_names, y_preds_val)):
        plt.scatter(y_true_val, pred, c=model_colors[i], alpha=0.7, s=marker_size, edgecolor=model_colors[i], linewidths=0.8, marker=markers[1], label=f'{name} (Val)')
    for i, (name, pred) in enumerate(zip(model_names, y_preds_test)):
        plt.scatter(y_true_test, pred, c=model_colors[i], alpha=0.7, s=marker_size, edgecolor=model_colors[i], linewidths=0.8, marker=markers[2], label=f'{name} (Test)')
    plt.xlim(min_limit, max_limit)
    plt.ylim(min_limit, max_limit)
    plt.xlabel('Experimental Ts (kNm)')
    plt.ylabel('Predicted Ts (kNm)')
    plt.title('Model Predictions Comparison')
    plt.legend(loc='lower right', ncol=2, framealpha=0.9, markerscale=1.2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(pad=3.0)
    plt.savefig('model_comparison.jpg', bbox_inches='tight')
    plt.show()


def plot_ensemble_weights(weights):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), dpi=150)
    model_names = ['PGNN', 'GNN', 'XGBoost']
    model_colors = [COLORS['pgnn'], COLORS['gnn'], COLORS['xgb']]
    model_weights = [weights['w1'], weights['w2'], weights['w3']]
    bars = ax1.bar(model_names, model_weights, color=model_colors)
    ax1.set_ylabel('Weight')
    ax1.set_title('Model Weight Contribution')
    ax1.set_ylim(0, 1)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    bar = ax2.bar(['Bias'], [weights['bias']], color=COLORS['bo_stack'])
    ax2.set_ylabel('Bias Value (kNm)')
    ax2.set_title('Bias Contribution')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    height = bar[0].get_height()
    ax2.text(bar[0].get_x() + bar[0].get_width()/2., height + (5 * np.sign(height)), f"{weights['bias']:.2f}", ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout(pad=1.5)
    plt.savefig('model_weights_bias.jpg', bbox_inches='tight')
    plt.show()

# ======================
# 7. Saving helpers
# ======================
def save_training_history(pgnn_trainer, gnn_trainer):
    pgnn_df = pd.DataFrame({'epoch': range(1, len(pgnn_trainer.train_losses)+1), 'train_loss': pgnn_trainer.train_losses, 'val_rmse': pgnn_trainer.val_losses})
    pgnn_df.to_csv('pgnn_training_history.csv', index=False)
    gnn_df = pd.DataFrame({'epoch': range(1, len(gnn_trainer.train_losses)+1), 'train_loss': gnn_trainer.train_losses, 'val_rmse': gnn_trainer.val_losses})
    gnn_df.to_csv('gnn_training_history.csv', index=False)
    print("Training history data saved to CSV files")


def save_residual_data(y_true, y_pred, set_name):
    residuals = y_true - y_pred
    residual_df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred, 'Residual': residuals, 'Set': set_name})
    return residual_df


def save_model_comparison_data(y_true_train, y_preds_train, y_true_val, y_preds_val, y_true_test, y_preds_test, model_names):
    comparison_data = []
    for i, name in enumerate(model_names):
        for j in range(len(y_true_train)):
            comparison_data.append({'Model': name, 'Set': 'Train', 'Actual': y_true_train[j], 'Predicted': y_preds_train[i][j]})
    for i, name in enumerate(model_names):
        for j in range(len(y_true_val)):
            comparison_data.append({'Model': name, 'Set': 'Validation', 'Actual': y_true_val[j], 'Predicted': y_preds_val[i][j]})
    for i, name in enumerate(model_names):
        for j in range(len(y_true_test)):
            comparison_data.append({'Model': name, 'Set': 'Test', 'Actual': y_true_test[j], 'Predicted': y_preds_test[i][j]})
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv('model_comparison_data.csv', index=False)
    print("Model comparison data saved to CSV")


def save_ensemble_weights(weights):
    weights_df = pd.DataFrame({'Model': ['PGNN', 'GNN', 'XGBoost', 'Bias'], 'Weight': [weights['w1'], weights['w2'], weights['w3'], weights['bias']]})
    weights_df.to_csv('ensemble_weights.csv', index=False)
    print("Ensemble weights saved to CSV")

# ======================
# 8. Main Pipeline
# ======================

def main():
    print("Loading and preprocessing data...")
    data = load_and_preprocess_data()
    print(f"Dataset sizes: Train={len(data['X_train'])}, Val={len(data['X_val'])}, Test={len(data['X_test'])}")

    # PGNN
    print("\nTraining Physics-Guided Neural Network (PGNN)...")
    pgnn_trainer = PGNN_Trainer(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        data['T_phy_train'], data['T_phy_val'],
        data['T_cap_train'], data['T_cap_val'],
        lam_data=1.0, lam_phy=0.3, lam_code=0.3
    )
    pgnn_trainer.train(epochs=200)

    # GNN
    print("\nTraining Graph Neural Network (GNN)...")
    gnn_trainer = GNN_Trainer(data['X_train_graph'], data['y_train'], data['X_val_graph'], data['y_val'])
    gnn_trainer.train(epochs=200)

    # XGBoost
    print("\nTraining XGBoost Model...")
    xgb_trainer = XGBoost_Trainer(data['X_train'], data['y_train'], data['X_val'], data['y_val'])
    xgb_trainer.train()

    # Validation predictions for stacking
    print("\nGenerating validation predictions...")
    pgnn_val_pred = pgnn_trainer.predict(data['X_val'])
    gnn_val_pred = gnn_trainer.predict(data['X_val_graph'])
    xgb_val_pred = xgb_trainer.predict(data['X_val'])

    # BO-Stack
    print("\nOptimizing ensemble weights with Bayesian Optimization...")
    bo_stack = BO_Stack(pgnn_val_pred, gnn_val_pred, xgb_val_pred, data['y_val'])
    weights = bo_stack.optimize_weights()
    print(f"\nOptimized Weights: PGNN={weights['w1']:.4f}, GNN={weights['w2']:.4f}, XGBoost={weights['w3']:.4f}, Bias={weights['bias']:.2f} kNm")

    # Predictions
    print("\nGenerating predictions...")
    pgnn_train_pred = pgnn_trainer.predict(data['X_train'])
    gnn_train_pred = gnn_trainer.predict(data['X_train_graph'])
    xgb_train_pred = xgb_trainer.predict(data['X_train'])
    ensemble_train_pred = bo_stack.predict(pgnn_train_pred, gnn_train_pred, xgb_train_pred)

    ensemble_val_pred = bo_stack.predict(pgnn_val_pred, gnn_val_pred, xgb_val_pred)

    pgnn_test_pred = pgnn_trainer.predict(data['X_test'])
    gnn_test_pred = gnn_trainer.predict(data['X_test_graph'])
    xgb_test_pred = xgb_trainer.predict(data['X_test'])
    ensemble_test_pred = bo_stack.predict(pgnn_test_pred, gnn_test_pred, xgb_test_pred)

    # Evaluate
    metrics_train, metrics_val, metrics_test = {}, {}, {}
    print("\n===== Training Set Evaluation =====")
    for name, pred in zip(['PGNN', 'GNN', 'XGBoost', 'BO-Stack'], [pgnn_train_pred, gnn_train_pred, xgb_train_pred, ensemble_train_pred]):
        metrics_train[name] = evaluate_model(data['y_train'], pred, name, 'Train')
    print("\n===== Validation Set Evaluation =====")
    for name, pred in zip(['PGNN', 'GNN', 'XGBoost', 'BO-Stack'], [pgnn_val_pred, gnn_val_pred, xgb_val_pred, ensemble_val_pred]):
        metrics_val[name] = evaluate_model(data['y_val'], pred, name, 'Val')
    print("\n===== Testing Set Evaluation =====")
    for name, pred in zip(['PGNN', 'GNN', 'XGBoost', 'BO-Stack'], [pgnn_test_pred, gnn_test_pred, xgb_test_pred, ensemble_test_pred]):
        metrics_test[name] = evaluate_model(data['y_test'], pred, name, 'Test')

    # Visualizations
    plot_bo_residuals(data['y_train'], ensemble_train_pred, data['y_val'], ensemble_val_pred, data['y_test'], ensemble_test_pred)
    plot_combined_training_history(pgnn_trainer, gnn_trainer)
    plot_model_comparison(
        data['y_train'], [pgnn_train_pred, gnn_train_pred, xgb_train_pred, ensemble_train_pred],
        data['y_val'], [pgnn_val_pred, gnn_val_pred, xgb_val_pred, ensemble_val_pred],
        data['y_test'], [pgnn_test_pred, gnn_test_pred, xgb_test_pred, ensemble_test_pred],
        ['PGNN', 'GNN', 'XGBoost', 'BO-Stack']
    )
    plot_ensemble_weights(weights)

    # Save CSVs
    print("\nSaving results to CSV files...")
    save_training_history(pgnn_trainer, gnn_trainer)
    ensemble_residuals = pd.concat([
        save_residual_data(data['y_train'], ensemble_train_pred, 'Train'),
        save_residual_data(data['y_val'], ensemble_val_pred, 'Validation'),
        save_residual_data(data['y_test'], ensemble_test_pred, 'Test')
    ])
    ensemble_residuals.to_csv('ensemble_residuals.csv', index=False)
    save_model_comparison_data(
        data['y_train'], [pgnn_train_pred, gnn_train_pred, xgb_train_pred, ensemble_train_pred],
        data['y_val'], [pgnn_val_pred, gnn_val_pred, xgb_val_pred, ensemble_val_pred],
        data['y_test'], [pgnn_test_pred, gnn_test_pred, xgb_test_pred, ensemble_test_pred],
        ['PGNN', 'GNN', 'XGBoost', 'BO-Stack']
    )
    save_ensemble_weights(weights)

    results_df = pd.DataFrame({
        'Actual': np.concatenate([data['y_train'], data['y_val'], data['y_test']]),
        'PGNN_Predicted': np.concatenate([pgnn_train_pred, pgnn_val_pred, pgnn_test_pred]),
        'GNN_Predicted': np.concatenate([gnn_train_pred, gnn_val_pred, gnn_test_pred]),
        'XGBoost_Predicted': np.concatenate([xgb_train_pred, xgb_val_pred, xgb_test_pred]),
        'Ensemble_Predicted': np.concatenate([ensemble_train_pred, ensemble_val_pred, ensemble_test_pred]),
        'Set': ['Train'] * len(data['y_train']) + ['Val'] * len(data['y_val']) + ['Test'] * len(data['y_test'])
    })
    results_df.to_csv('bo_stack_predictions.csv', index=False)

    metrics_data = []
    for model in ['PGNN', 'GNN', 'XGBoost', 'BO-Stack']:
        for set_name, pack in [('Train', metrics_train), ('Val', metrics_val), ('Test', metrics_test)]:
            rmse, mae, r2, bias = pack[model]
            metrics_data.append({'Model': model, 'Set': set_name, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'Bias': bias})
    pd.DataFrame(metrics_data).to_csv('model_metrics.csv', index=False)

    joblib.dump(data['scaler_X'], 'scaler_X.pkl')
    print("\nResults saved to CSV files")
    print("Visualizations saved as JPG files")
    print("Scalers saved as pickle files")


if __name__ == '__main__':
    main()