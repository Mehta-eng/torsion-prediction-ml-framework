PGNN for RC Torsion — Code

Header styles supported:
  1) Index-form (preferred, EXACT theory units):
     b,h,b_core,h_core,fc,ALfyl,Atfyt_per_s,Ts
        - ALfyl in kN
        - Atfyt_per_s in N/mm
        - Ts in kN·m
  2) Classic engineering:
     b,h,fc,cover,fyl,fyt,Al,Ts
  3) Generic:
     X1..X7 + Y8  (mapped to classic order)

Outputs (Desktop):
  - training_history_improved.csv  (Train/Val/Test loss per epoch)
  - loss_components_improved.csv   (Data/Physics/Code per epoch)
  - predictions_residuals_improved.csv  (includes T_phy, T_max)
  - model_metrics_improved.csv
  - rmse_evolution_improved.csv
  - violations_summary.csv
  - anchors_correlations.csv
  - learned_params.csv
  - train_pred_vs_actual.csv, val_pred_vs_actual.csv, test_pred_vs_actual.csv
  - improved_results.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -----------------------------
# Hyperparameters
# -----------------------------
HIDDEN_LAYERS = [256, 128, 64]
DROPOUT_RATE = 0.2
ACTIVATION = tf.nn.silu
EPOCHS = 2000
LR = 5e-4
PATIENCE = 200

# Loss weights (θ fixed per theory)
LAMBDA_DATA = 1.0
LAMBDA_PHYSICS_INIT = 0.10
LAMBDA_PHYSICS_FINAL = 0.50
LAMBDA_CODE = 0.10
THETA_DEG_FIXED = 45.0  # theory

OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Desktop")
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# -----------------------------
# Utilities
# -----------------------------
def winsorize_df(df, lower_q=0.01, upper_q=0.99):
    df = df.copy()
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.number):
            lo, hi = df[c].quantile([lower_q, upper_q])
            df[c] = df[c].clip(lo, hi)
    return df


def detect_column_mapping(df):
    cols = set(df.columns.str.strip())

    # Index-form (theory-aligned) dataset
    if {'b','h','b_core','h_core','fc','ALfyl','Atfyt_per_s','Ts'}.issubset(cols):
        return (['b','h','b_core','h_core','fc','ALfyl','Atfyt_per_s'], 'Ts', 'index')

    # Classic engineering dataset
    if {'b','h','fc','cover','fyl','fyt','Al','Ts'}.issubset(cols):
        return (['b','h','fc','cover','fyl','fyt','Al'], 'Ts', 'classic')

    # Generic X1..X7 + Y8 (mapped to classic order)
    x_feats = [f'X{i}' for i in range(1,8)]
    if set(x_feats).issubset(cols) and 'Y8' in cols:
        return (x_feats, 'Y8', 'generic')

    raise ValueError(
        "Unrecognized CSV headers.\n"
        "Expected either:\n"
        "  (a) index-form: b,h,b_core,h_core,fc,ALfyl,Atfyt_per_s,Ts\n"
        "  (b) classic:    b,h,fc,cover,fyl,fyt,Al,Ts\n"
        "  (c) generic:    X1..X7 + Y8\n"
        f"Found: {df.columns.tolist()}"
    )


def remap_to_engineering_order(df, feat_names, mode):
    if mode == 'index':
        mapping = ['b','h','b_core','h_core','fc','ALfyl','Atfyt_per_s']
    elif mode == 'classic':
        mapping = ['b','h','fc','cover','fyl','fyt','Al']
    else:  # generic -> classic mapping
        mapping = ['X1','X2','X3','X4','X5','X6','X7']
    return df[mapping].values.astype(np.float32)


def create_interaction_features(X, mode):
    """
    Build enhanced features. Always returns 14 cols for consistent MLP input size.
    """
    if mode == 'index':
        b, h, b_core, h_core, f_c, ALfyl_kN, Atfyt_s = (X[:,0],X[:,1],X[:,2],X[:,3],X[:,4],X[:,5],X[:,6])
        A_core = b_core * h_core
        p_h = 2.0 * (b_core + h_core)
        A0 = 0.85 * A_core
        enh = np.stack([
            b, h, b_core, h_core, f_c, ALfyl_kN, Atfyt_s,
            A_core, p_h, A0, b/np.maximum(h,1e-6), b_core/np.maximum(h_core,1e-6),
            (ALfyl_kN * p_h), (Atfyt_s * A0)
        ], axis=1).astype(np.float32)
        return enh

    # classic/generic
    b, h, f_c, cover, f_yl, f_yt, A_l = (X[:,0],X[:,1],X[:,2],X[:,3],X[:,4],X[:,5],X[:,6])
    aspect_ratio = b / np.maximum(h, 1e-6)
    area = b * h
    reinforcement_ratio = A_l / np.maximum(area, 1e-6)
    cover_ratio = cover / np.maximum(np.minimum(b, h), 1e-6)
    enh = np.stack([
        b, h, f_c, cover, f_yl, f_yt, A_l,
        aspect_ratio, area, reinforcement_ratio, cover_ratio,
        b * f_c, h * f_c, A_l * f_yl
    ], axis=1).astype(np.float32)
    return enh


# -----------------------------
# Model
# -----------------------------
class HybridTorsionModel(tf.keras.Model):
    def __init__(self, mode, hidden_layers=HIDDEN_LAYERS, activation=ACTIVATION,
                 dropout_rate=DROPOUT_RATE, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode  # 'index' | 'classic' | 'generic->classic'
        self.dense_layers = []
        for units in hidden_layers:
            self.dense_layers.append(tf.keras.layers.Dense(units, activation=activation))
            self.dense_layers.append(tf.keras.layers.BatchNormalization())
            self.dense_layers.append(tf.keras.layers.Dropout(dropout_rate))
        self.output_layer = tf.keras.layers.Dense(1, activation='linear')

        # Legacy learnables (used only in classic mode)
        self.A_t = tf.Variable(50.0, trainable=True, dtype=tf.float32, name="A_t")     # mm^2
        self.s   = tf.Variable(150.0, trainable=True, dtype=tf.float32, name="s")      # mm

        # Fixed physics weight (annealed externally) and fixed θ
        self.lambda_physics = tf.Variable(LAMBDA_PHYSICS_INIT, trainable=False, dtype=tf.float32)
        self.theta_deg = tf.constant(THETA_DEG_FIXED, dtype=tf.float32)  # fixed per theory

    def call(self, x_scaled, training=False):
        z = x_scaled
        for layer in self.dense_layers:
            if isinstance(layer, (tf.keras.layers.Dropout, tf.keras.layers.BatchNormalization)):
                z = layer(z, training=training)
            else:
                z = layer(z)
        return self.output_layer(z)

    def calculate_physics_based_torsion(self, X_raw):
        """
        Returns (T_phy, T_max) in kN·m using the detected mode.
        INDEX mode (theory-aligned):
          X_raw = [b,h,b_core,h_core,fc,ALfyl(kN),Atfyt/s(N/mm)]
        """
        if self.mode == 'index':
            b       = X_raw[:,0]; h       = X_raw[:,1]
            b_core  = tf.maximum(X_raw[:,2], 1.0)
            h_core  = tf.maximum(X_raw[:,3], 1.0)
            f_c     = X_raw[:,4]
            ALfyl_kN     = X_raw[:,5]     # kN
            Atfyt_s      = X_raw[:,6]     # N/mm

            A_core = b_core * h_core
            A0     = 0.85 * A_core
            p_h    = 2.0 * (b_core + h_core)

            theta_rad = self.theta_deg * (np.pi / 180.0)
            cot_theta = 1.0 / tf.tan(theta_rad)
            tan_theta = tf.tan(theta_rad)
            sqrt_fc   = tf.sqrt(tf.maximum(f_c, 1.0))

            # N·mm → kN·m (1e-6) ; kN·mm → kN·m (1e-3)
            T_c   = 0.33 * sqrt_fc * (A_core ** 2) / tf.maximum(p_h, 1.0) * 1e-6
            T_s   = 2.0 * A0 * Atfyt_s * cot_theta * 1e-6
            T_L   = (ALfyl_kN * p_h) / 2.0 * tan_theta * 1e-3
            T_max = 0.20 * sqrt_fc * (A_core ** 2) / tf.maximum(p_h, 1.0) * 1e-6

            T_phy = T_c + T_s + T_L
            return tf.reshape(T_phy, [-1,1]), tf.reshape(T_max, [-1,1])

        # -------- Classic/generic fallback (uses learnable A_t, s) --------
        b = X_raw[:,0]; h = X_raw[:,1]; f_c = X_raw[:,2]; cover = X_raw[:,3]
        f_yl = X_raw[:,4]; f_yt = X_raw[:,5]; A_l = X_raw[:,6]
        A_t = tf.maximum(self.A_t, 10.0)
        s   = tf.maximum(self.s,   50.0)

        b_core = tf.maximum(b - 2.0*cover, 1.0)
        h_core = tf.maximum(h - 2.0*cover, 1.0)
        A_core = b_core * h_core
        A0     = 0.85 * A_core
        p_h    = 2.0 * (b_core + h_core)

        theta_rad = self.theta_deg * (np.pi/180.0)
        cot_theta = 1.0 / tf.tan(theta_rad)
        tan_theta = tf.tan(theta_rad)
        sqrt_fc   = tf.sqrt(tf.maximum(f_c, 1.0))

        T_c   = 0.33 * sqrt_fc * (A_core ** 2) / tf.maximum(p_h, 1.0) * 1e-6
        T_s   = 2.0 * A0 * (A_t * f_yt) / tf.maximum(s, 1.0) * cot_theta * 1e-6
        T_L   = (A_l * f_yl * p_h) / 2.0 * tan_theta * 1e-6   # no "/ s" per theory
        T_max = 0.20 * sqrt_fc * (A_core ** 2) / tf.maximum(p_h, 1.0) * 1e-6

        T_phy = T_c + T_s + T_L
        return tf.reshape(T_phy, [-1,1]), tf.reshape(T_max, [-1,1])

    def physics_guided_loss(self, y_true_scaled, y_pred_scaled, X_raw, y_scaler):
        T_phy, T_max = self.calculate_physics_based_torsion(X_raw)

        # scale anchors into y-scaled space
        y_mean  = tf.constant(y_scaler.mean_,  dtype=tf.float32)
        y_scale = tf.constant(y_scaler.scale_, dtype=tf.float32)
        T_phy_scaled = (T_phy - y_mean) / y_scale
        T_max_scaled = (T_max - y_mean) / y_scale

        data_loss    = tf.reduce_mean(tf.square(y_pred_scaled - y_true_scaled))
        physics_loss = tf.reduce_mean(tf.square(y_pred_scaled - T_phy_scaled))
        code_loss    = tf.reduce_mean(tf.square(tf.maximum(y_pred_scaled - T_max_scaled, 0.0)))

        total_loss = (LAMBDA_DATA * data_loss +
                      self.lambda_physics * physics_loss +
                      LAMBDA_CODE * code_loss)
        return total_loss, data_loss, physics_loss, code_loss


# -----------------------------
# Training loop
# -----------------------------
def train_model(model,
                Xtr_scaled, ytr_scaled,
                Xva_scaled, yva_scaled,
                Xte_scaled, yte_scaled,
                Xtr_raw, Xva_raw, Xte_raw,
                y_scaler,
                epochs=EPOCHS, lr=LR, patience=PATIENCE):

    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    best_val = np.inf
    wait = 0
    weights_path = 'best_model.weights.h5'

    tr_hist, va_hist, te_hist = [], [], []
    data_hist, phys_hist, code_hist = [], [], []
    tr_rmse_hist, va_rmse_hist, te_rmse_hist = [], [], []
    lam_hist = []

    for ep in range(epochs):
        # cosine anneal λ_phys
        progress = ep / max(1, epochs - 1)
        lam = LAMBDA_PHYSICS_INIT + (LAMBDA_PHYSICS_FINAL - LAMBDA_PHYSICS_INIT) * 0.5 * (1 - np.cos(np.pi * progress))
        model.lambda_physics.assign(lam)
        lam_hist.append(float(lam))

        # ---- train ----
        with tf.GradientTape() as tape:
            y_pred_tr = model(Xtr_scaled, training=True)
            loss, dL, pL, cL = model.physics_guided_loss(ytr_scaled, y_pred_tr, Xtr_raw, y_scaler)
        grads = tape.gradient(loss, model.trainable_variables)
        grads = [tf.clip_by_value(g, -1.0, 1.0) for g in grads if g is not None]
        opt.apply_gradients(zip(grads, model.trainable_variables))

        # ---- val/test (eval only) ----
        y_pred_va = model(Xva_scaled, training=False)
        val_loss, _, _, _ = model.physics_guided_loss(yva_scaled, y_pred_va, Xva_raw, y_scaler)

        y_pred_te = model(Xte_scaled, training=False)
        test_loss, _, _, _ = model.physics_guided_loss(yte_scaled, y_pred_te, Xte_raw, y_scaler)

        tr_hist.append(float(loss.numpy()))
        va_hist.append(float(val_loss.numpy()))
        te_hist.append(float(test_loss.numpy()))
        data_hist.append(float(dL.numpy()))
        phys_hist.append(float(pL.numpy()))
        code_hist.append(float(cL.numpy()))

        # RMSE snapshots (scaled space; trend only)
        if ep % 10 == 0:
            tr_rmse = float(np.sqrt(mean_squared_error(ytr_scaled.numpy(), y_pred_tr.numpy())))
            va_rmse = float(np.sqrt(mean_squared_error(yva_scaled.numpy(), y_pred_va.numpy())))
            te_rmse = float(np.sqrt(mean_squared_error(yte_scaled.numpy(), y_pred_te.numpy())))
            tr_rmse_hist.append((ep, tr_rmse))
            va_rmse_hist.append((ep, va_rmse))
            te_rmse_hist.append((ep, te_rmse))

        # early stopping based on validation
        if val_loss < best_val - 1e-8:
            best_val = float(val_loss.numpy())
            wait = 0
            model.save_weights(weights_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {ep}")
                break

        if ep % 100 == 0:
            print(f"Epoch {ep:04d} | Train {tr_hist[-1]:.4f} | Val {va_hist[-1]:.4f} | Test {te_hist[-1]:.4f}")
            print(f"  Data {data_hist[-1]:.4f} | Phys {phys_hist[-1]:.4f} | Code {code_hist[-1]:.4f} | λ_phys {lam_hist[-1]:.4f}")

    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print("Loaded best model weights.")

    return (tr_hist, va_hist, te_hist,
            data_hist, phys_hist, code_hist,
            tr_rmse_hist, va_rmse_hist, te_rmse_hist,
            lam_hist)


# -----------------------------
# Main
# -----------------------------
def main():
    # --- Load ---
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = r"C:\Users\vikas\Desktop\Dataset.csv"
    df = pd.read_csv(csv_path)

    # detect headers and mode
    feat_names, target_name, mode = detect_column_mapping(df)

    # winsorize numeric columns
    df = winsorize_df(df, 0.01, 0.99)

    # raw features and target
    X_raw_np = remap_to_engineering_order(df, feat_names, mode)
    y_np = df[target_name].values.reshape(-1,1).astype(np.float32)

    # enhanced features for MLP
    X_enhanced = create_interaction_features(X_raw_np, mode)

    # split (stratify on target deciles if possible)
    try:
        bins = pd.qcut(df[target_name], q=10, labels=False, duplicates='drop')
    except Exception:
        bins = pd.cut(df[target_name], bins=10, labels=False)

    X_train_val, X_test, y_train_val, y_test, X_raw_train_val, X_raw_test, bins_train_val, _ = train_test_split(
        X_enhanced, y_np, X_raw_np, bins, test_size=0.2, random_state=SEED, stratify=bins
    )
    X_train, X_val, y_train, y_val, X_raw_train, X_raw_val, _, _ = train_test_split(
        X_train_val, y_train_val, X_raw_train_val, bins_train_val, test_size=0.2, random_state=SEED, stratify=bins_train_val
    )

    print(f"Training set: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")
    print(f"Feature dims (enhanced): {X_train.shape[1]}")

    # scale
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    Xtr_s = x_scaler.fit_transform(X_train).astype(np.float32)
    Xva_s = x_scaler.transform(X_val).astype(np.float32)
    Xte_s = x_scaler.transform(X_test).astype(np.float32)

    ytr_s = y_scaler.fit_transform(y_train).astype(np.float32)
    yva_s = y_scaler.transform(y_val).astype(np.float32)
    yte_s = y_scaler.transform(y_test).astype(np.float32)

    # tensors
    Xtr_s_tf = tf.convert_to_tensor(Xtr_s, dtype=tf.float32)
    Xva_s_tf = tf.convert_to_tensor(Xva_s, dtype=tf.float32)
    Xte_s_tf = tf.convert_to_tensor(Xte_s, dtype=tf.float32)

    ytr_tf = tf.convert_to_tensor(ytr_s, dtype=tf.float32)
    yva_tf = tf.convert_to_tensor(yva_s, dtype=tf.float32)
    yte_tf = tf.convert_to_tensor(yte_s, dtype=tf.float32)

    Xtr_raw_tf = tf.convert_to_tensor(X_raw_train, dtype=tf.float32)
    Xva_raw_tf = tf.convert_to_tensor(X_raw_val,   dtype=tf.float32)
    Xte_raw_tf = tf.convert_to_tensor(X_raw_test,  dtype=tf.float32)

    # --- Train ---
    model = HybridTorsionModel(mode=mode)
    print("Training Physics-Guided Neural Network …")

    (tr_hist, va_hist, te_hist,
     data_hist, phys_hist, code_hist,
     tr_rmse_hist, va_rmse_hist, te_rmse_hist,
     lam_hist) = train_model(
        model,
        Xtr_s_tf, ytr_tf,
        Xva_s_tf, yva_tf,
        Xte_s_tf, yte_tf,
        Xtr_raw_tf, Xva_raw_tf, Xte_raw_tf,
        y_scaler,
        epochs=EPOCHS, lr=LR, patience=PATIENCE
    )

    # --- Predictions (back to original kN·m) ---
    def inv_scale(z): return y_scaler.inverse_transform(z)

    y_train_pred = inv_scale(model(Xtr_s_tf, training=False).numpy())
    y_val_pred   = inv_scale(model(Xva_s_tf, training=False).numpy())
    y_test_pred  = inv_scale(model(Xte_s_tf, training=False).numpy())

    y_train_true = inv_scale(ytr_s)
    y_val_true   = inv_scale(yva_s)
    y_test_true  = inv_scale(yte_s)

    # --- Metrics ---
    def metrics(y_true, y_pred, name):
        mse = mean_squared_error(y_true, y_pred); rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2  = r2_score(y_true, y_pred)
        print(f"\n{name} | MSE={mse:.4f}  RMSE={rmse:.4f} kN·m  MAE={mae:.4f} kN·m  R²={r2:.4f}")
        return mse, rmse, mae, r2

    train_m = metrics(y_train_true, y_train_pred, "Training")
    val_m   = metrics(y_val_true,   y_val_pred,   "Validation")
    test_m  = metrics(y_test_true,  y_test_pred,  "Test")

    # --- Physics anchors for each split (kN·m) ---
    def anchors_df(X_raw_tf, tag):
        T_phy, T_max = model.calculate_physics_based_torsion(X_raw_tf)
        return pd.DataFrame({'Dataset': tag,
                             'T_phy': T_phy.numpy().flatten(),
                             'T_max': T_max.numpy().flatten()})

    phy_tr = anchors_df(Xtr_raw_tf, 'Training')
    phy_va = anchors_df(Xva_raw_tf, 'Validation')
    phy_te = anchors_df(Xte_raw_tf, 'Test')
    phy_all = pd.concat([phy_tr, phy_va, phy_te], ignore_index=True)

    # --- CSV outputs ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Training history (totals + components + λ) — includes TEST loss per epoch
    pd.DataFrame({
        'Epoch': np.arange(1, len(tr_hist)+1, dtype=int),
        'Train_Loss': tr_hist,
        'Validation_Loss': va_hist,
        'Test_Loss': te_hist,
        'Data_Loss': data_hist,
        'Physics_Loss': phys_hist,
        'Code_Loss': code_hist,
        'lambda_physics': lam_hist
    }).to_csv(os.path.join(OUTPUT_DIR, 'training_history_improved.csv'), index=False)

    # Explicit loss-components CSV (same content; kept for clarity)
    pd.DataFrame({
        'Epoch': np.arange(1, len(tr_hist)+1, dtype=int),
        'Train_Loss_Total': tr_hist,
        'Validation_Loss_Total': va_hist,
        'Test_Loss_Total': te_hist,
        'Data_Loss': data_hist,
        'Physics_Loss': phys_hist,
        'Code_Loss': code_hist,
        'lambda_physics': lam_hist
    }).to_csv(os.path.join(OUTPUT_DIR, 'loss_components_improved.csv'), index=False)

    # 2) Combined predictions + residuals + anchors
    preds_df = pd.DataFrame({
        'Dataset': (['Training'] * len(y_train_true) +
                    ['Validation'] * len(y_val_true) +
                    ['Test'] * len(y_test_true)),
        'Actual_kN_m': np.concatenate([y_train_true.flatten(), y_val_true.flatten(), y_test_true.flatten()]),
        'Predicted_kN_m': np.concatenate([y_train_pred.flatten(), y_val_pred.flatten(), y_test_pred.flatten()])
    })
    preds_df['Residual_kN_m'] = preds_df['Actual_kN_m'] - preds_df['Predicted_kN_m']
    preds_df = pd.concat([preds_df.reset_index(drop=True), phy_all[['T_phy','T_max']]], axis=1)
    preds_df.to_csv(os.path.join(OUTPUT_DIR, 'predictions_residuals_improved.csv'), index=False)

    # 2b) Split-specific Actual vs Predicted CSVs (with anchors per split)
    T_phy_tr, T_max_tr = model.calculate_physics_based_torsion(Xtr_raw_tf)
    T_phy_va, T_max_va = model.calculate_physics_based_torsion(Xva_raw_tf)
    T_phy_te, T_max_te = model.calculate_physics_based_torsion(Xte_raw_tf)

    train_split_df = pd.DataFrame({
        'Actual_kN_m':    y_train_true.ravel(),
        'Predicted_kN_m': y_train_pred.ravel(),
        'Residual_kN_m':  (y_train_true - y_train_pred).ravel(),
        'T_phy_kN_m':     T_phy_tr.numpy().ravel(),
        'T_max_kN_m':     T_max_tr.numpy().ravel(),
    })
    train_split_df.to_csv(os.path.join(OUTPUT_DIR, 'train_pred_vs_actual.csv'), index=False)

    val_split_df = pd.DataFrame({
        'Actual_kN_m':    y_val_true.ravel(),
        'Predicted_kN_m': y_val_pred.ravel(),
        'Residual_kN_m':  (y_val_true - y_val_pred).ravel(),
        'T_phy_kN_m':     T_phy_va.numpy().ravel(),
        'T_max_kN_m':     T_max_va.numpy().ravel(),
    })
    val_split_df.to_csv(os.path.join(OUTPUT_DIR, 'val_pred_vs_actual.csv'), index=False)

    test_split_df = pd.DataFrame({
        'Actual_kN_m':    y_test_true.ravel(),
        'Predicted_kN_m': y_test_pred.ravel(),
        'Residual_kN_m':  (y_test_true - y_test_pred).ravel(),
        'T_phy_kN_m':     T_phy_te.numpy().ravel(),
        'T_max_kN_m':     T_max_te.numpy().ravel(),
    })
    test_split_df.to_csv(os.path.join(OUTPUT_DIR, 'test_pred_vs_actual.csv'), index=False)

    # 3) Metrics table
    pd.DataFrame({
        'Dataset': ['Training','Validation','Test'],
        'MSE':  [train_m[0], val_m[0], test_m[0]],
        'RMSE': [train_m[1], val_m[1], test_m[1]],
        'MAE':  [train_m[2], val_m[2], test_m[2]],
        'R2':   [train_m[3], val_m[3], test_m[3]]
    }).to_csv(os.path.join(OUTPUT_DIR, 'model_metrics_improved.csv'), index=False)

    # 4) RMSE evolution (scaled-space trend)
    pd.DataFrame({
        'Epoch': [ep for ep,_ in tr_rmse_hist] + [ep for ep,_ in va_rmse_hist] + [ep for ep,_ in te_rmse_hist],
        'RMSE_scaled': [rm for _,rm in tr_rmse_hist] + [rm for _,rm in va_rmse_hist] + [rm for _,rm in te_rmse_hist],
        'Dataset': (['Training'] * len(tr_rmse_hist) + ['Validation'] * len(va_rmse_hist) + ['Test'] * len(te_rmse_hist))
    }).to_csv(os.path.join(OUTPUT_DIR, 'rmse_evolution_improved.csv'), index=False)

    # 5) Violations summary
    preds_df['Violation'] = (preds_df['Predicted_kN_m'] > preds_df['T_max']).astype(int)
    viol = preds_df.groupby('Dataset', observed=False).agg(
        n=('Violation','size'),
        violations=('Violation','sum')
    ).reset_index()
    viol['violation_rate_%'] = 100.0 * viol['violations'] / viol['n']
    viol.to_csv(os.path.join(OUTPUT_DIR, 'violations_summary.csv'), index=False)

    # 6) Anchor correlations (Pearson/Spearman)
    anchors_corr = pd.DataFrame({
        'pearson(Predicted,T_phy)': [preds_df['Predicted_kN_m'].corr(preds_df['T_phy'], method='pearson')],
        'pearson(Predicted,T_max)': [preds_df['Predicted_kN_m'].corr(preds_df['T_max'], method='pearson')],
        'pearson(Actual,T_phy)':    [preds_df['Actual_kN_m'].corr(preds_df['T_phy'], method='pearson')],
        'pearson(Actual,T_max)':    [preds_df['Actual_kN_m'].corr(preds_df['T_max'], method='pearson')],
        'spearman(Predicted,T_phy)':[preds_df['Predicted_kN_m'].corr(preds_df['T_phy'], method='spearman')],
        'spearman(Predicted,T_max)':[preds_df['Predicted_kN_m'].corr(preds_df['T_max'], method='spearman')],
        'spearman(Actual,T_phy)':   [preds_df['Actual_kN_m'].corr(preds_df['T_phy'], method='spearman')],
        'spearman(Actual,T_max)':   [preds_df['Actual_kN_m'].corr(preds_df['T_max'], method='spearman')],
    })
    anchors_corr.to_csv(os.path.join(OUTPUT_DIR, 'anchors_correlations.csv'), index=False)

    # 7) Learned params (A_t,s only relevant in classic mode)
    learned = {
        'theta_deg_fixed': THETA_DEG_FIXED,
        'A_t_mm2_(classic_only)': float(model.A_t.numpy()),
        's_mm_(classic_only)': float(model.s.numpy())
    }
    pd.DataFrame([learned]).to_csv(os.path.join(OUTPUT_DIR, 'learned_params.csv'), index=False)

    # --- Plot summary ---
    plt.figure(figsize=(18, 10))
    # Training history (three totals)
    plt.subplot(2,3,1)
    plt.plot(tr_hist, label='Train'); plt.plot(va_hist, label='Val'); plt.plot(te_hist, label='Test')
    plt.yscale('log'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training History'); plt.legend()

    # Loss components
    plt.subplot(2,3,2)
    plt.plot(data_hist, label='Data'); plt.plot(phys_hist, label='Physics'); plt.plot(code_hist, label='Code')
    plt.yscale('log'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Components'); plt.legend()

    # Preds vs Actual
    plt.subplot(2,3,3)
    plt.scatter(y_train_true, y_train_pred, s=12, alpha=0.6, label='Train')
    plt.scatter(y_val_true,   y_val_pred,   s=12, alpha=0.6, label='Val')
    plt.scatter(y_test_true,  y_test_pred,  s=12, alpha=0.6, label='Test')
    allv = np.concatenate([y_train_true, y_val_true, y_test_true]); mn, mx = allv.min(), allv.max()
    plt.plot([mn,mx],[mn,mx],'k--'); plt.xlabel('Actual (kN·m)'); plt.ylabel('Pred (kN·m)'); plt.title('Pred vs Actual'); plt.legend()

    # Residual hist
    plt.subplot(2,3,4)
    plt.hist((y_train_true - y_train_pred), bins=30, alpha=0.5, label='Train')
    plt.hist((y_val_true   - y_val_pred),   bins=30, alpha=0.5, label='Val')
    plt.hist((y_test_true  - y_test_pred),  bins=30, alpha=0.5, label='Test')
    plt.xlabel('Residual (kN·m)'); plt.ylabel('Count'); plt.title('Residuals'); plt.legend()

    # R² bars
    plt.subplot(2,3,5)
    r2s = [r2_score(y_train_true, y_train_pred), r2_score(y_val_true, y_val_pred), r2_score(y_test_true, y_test_pred)]
    plt.bar(['Train','Val','Test'], r2s); plt.ylim(0,1)
    for i,v in enumerate(r2s): plt.text(i, v+0.01, f'{v:.3f}', ha='center')
    plt.ylabel('R²'); plt.title('Performance')

    # RMSE evolution (scaled)
    plt.subplot(2,3,6)
    if tr_rmse_hist: tr_ep,tr_rm = zip(*tr_rmse_hist); plt.plot(tr_ep,tr_rm,label='Train')
    if va_rmse_hist: va_ep,va_rm = zip(*va_rmse_hist); plt.plot(va_ep,va_rm,label='Val')
    if te_rmse_hist: te_ep,te_rm = zip(*te_rmse_hist); plt.plot(te_ep,te_rm,label='Test')
    plt.xlabel('Epoch'); plt.ylabel('RMSE (scaled)'); plt.title('RMSE Evolution'); plt.legend(); plt.grid(True, ls='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'improved_results.png'), dpi=200)
    plt.close()

    print(f"\nAll CSVs and plots saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
