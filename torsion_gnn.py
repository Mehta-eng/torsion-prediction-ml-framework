PGNN for RC Torsion — Revised (2nd revision)

Expected CSV columns:

    b, h, b', h', fc, ALfyl, Atfyt/s, Ts

Recommended header names:
    b,h,b_core,h_core,fc,ALfyl,Atfyt_per_s,Ts

The script also accepts:
    b,h,b',h',fc,ALfyl,Atfyt/s,Ts

Implemented manuscript equations:
    Eq. S1:
        T_n,phy = min[
            2 A0 (At fyt / s) cot(theta),
            2 A0 (AL fyl) / ph tan(theta)
        ]

    Eq. S2:
        T_cr = 0.33 sqrt(fc) A_core^2 / ph

    Eq. S3:
        L_data = MSE(T_pred, T_exp)

    Eq. S4:
        L_physics = MSE(T_pred, T_n,phy)

    Eq. S5:
        L_upper = MSE(max(0, T_pred - T_max)),
        T_max = alpha T_n,phy

    Eq. S6:
        L_total = lambda_data L_data
                + lambda_physics L_physics
                + lambda_upper L_upper

    Eq. S7:
        phi_ML = exp(mu_epsilon - 1.645 sigma_epsilon)

Run:
    py -3.11 torsion_pgnn.py
or:
    py -3.11 torsion_pgnn.py "C:\\Users\\owner\\Desktop\\PGNN\\Dataset.csv"
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Neural network
HIDDEN_LAYERS = [256, 128, 64]
DROPOUT_RATE = 0.20
ACTIVATION = tf.nn.silu

# Training
EPOCHS = 2000
LR = 5e-4
PATIENCE = 200
GRAD_CLIP = 1.0

# Loss weights
LAMBDA_DATA = 1.0
LAMBDA_PHYSICS_INIT = 0.10
LAMBDA_PHYSICS_FINAL = 0.50
LAMBDA_UPPER = 0.10

# Physics constants
THETA_DEG_FIXED = 45.0
ALPHA_UPPER = 1.20
A0_FACTOR = 0.85
CRACKING_COEFF = 0.33

# Split
TEST_SIZE = 0.20
VAL_SIZE_OF_REMAINING = 0.20

# Output
OUTPUT_DIR = Path.home() / "Desktop" / "PGNN_outputs_8col"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Force-map the exact 8-column dataset into:
        b, h, b_core, h_core, fc, ALfyl, Atfyt_per_s, Ts

    Your CSV header may be:
        b, h, b', h', fc, ALfyl, Atfyt/s, Ts

    This function intentionally ignores symbolic header differences and maps
    columns by their exact order to avoid KeyError problems caused by b' and h'.
    """
    if df.shape[1] != 8:
        raise ValueError(
            f"Expected exactly 8 columns: b, h, b', h', fc, ALfyl, Atfyt/s, Ts. "
            f"Found {df.shape[1]} columns."
        )

    df = df.copy()
    df.columns = ["b", "h", "b_core", "h_core", "fc", "ALfyl", "Atfyt_per_s", "Ts"]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)

    if after < before:
        print(f"Removed {before - after} rows containing non-numeric or missing values.")

    if df.empty:
        raise ValueError("No numeric data remained after cleaning the CSV.")

    return df


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Load the exact 8-column CSV.

    Expected order:
        b, h, b', h', fc, ALfyl, Atfyt/s, Ts

    The file may contain a header row. If the header row is missing, the script
    still assigns the same 8-column order.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        raise RuntimeError(f"Could not read CSV file: {csv_path}") from exc

    # If pandas treated the first data row as headers, most column names will
    # look numeric. Then reload with header=None.
    numeric_header_count = 0
    for col in df.columns:
        try:
            float(str(col))
            numeric_header_count += 1
        except ValueError:
            pass

    if numeric_header_count >= 4:
        print("Numeric-looking header detected. Re-reading CSV using header=None.")
        df = pd.read_csv(csv_path, header=None)

    print("Original CSV columns:", list(df.columns))
    df = normalize_columns(df)
    print("Mapped columns:", list(df.columns))

    return df


def winsorize_train_only(X_train, y_train, lower=0.01, upper=0.99):
    """Winsorize training data only to avoid validation/test leakage."""
    Xw = X_train.copy()
    yw = y_train.copy()

    for j in range(Xw.shape[1]):
        lo, hi = np.quantile(Xw[:, j], [lower, upper])
        Xw[:, j] = np.clip(Xw[:, j], lo, hi)

    lo, hi = np.quantile(yw[:, 0], [lower, upper])
    yw[:, 0] = np.clip(yw[:, 0], lo, hi)

    return Xw.astype(np.float32), yw.astype(np.float32)


def make_enhanced_features(X_raw: np.ndarray) -> np.ndarray:
    """
    Build enhanced input features from the exact 8-column raw data.

    Raw order:
        b, h, b_core, h_core, fc, ALfyl, Atfyt_per_s
    """
    b = X_raw[:, 0]
    h = X_raw[:, 1]
    b_core = X_raw[:, 2]
    h_core = X_raw[:, 3]
    fc = X_raw[:, 4]
    ALfyl = X_raw[:, 5]
    Atfyt_per_s = X_raw[:, 6]

    A_core = b_core * h_core
    A0 = A0_FACTOR * A_core
    ph = 2.0 * (b_core + h_core)

    aspect = b / np.maximum(h, 1e-6)
    core_aspect = b_core / np.maximum(h_core, 1e-6)
    sqrt_fc = np.sqrt(np.maximum(fc, 1e-6))

    enhanced = np.stack(
        [
            b,
            h,
            b_core,
            h_core,
            fc,
            ALfyl,
            Atfyt_per_s,
            A_core,
            A0,
            ph,
            aspect,
            core_aspect,
            sqrt_fc,
            ALfyl * A0 / np.maximum(ph, 1e-6),
            Atfyt_per_s * A0,
        ],
        axis=1,
    ).astype(np.float32)

    return enhanced


# ============================================================
# MODEL
# ============================================================

class PhysicsGuidedTorsionModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.layers_list = []
        for units in HIDDEN_LAYERS:
            self.layers_list.append(tf.keras.layers.Dense(units, activation=ACTIVATION))
            self.layers_list.append(tf.keras.layers.BatchNormalization())
            self.layers_list.append(tf.keras.layers.Dropout(DROPOUT_RATE))

        self.output_layer = tf.keras.layers.Dense(1, activation="linear")

        self.lambda_physics = tf.Variable(
            LAMBDA_PHYSICS_INIT, trainable=False, dtype=tf.float32
        )

    def call(self, x_scaled, training=False):
        z = x_scaled
        for layer in self.layers_list:
            if isinstance(layer, (tf.keras.layers.BatchNormalization, tf.keras.layers.Dropout)):
                z = layer(z, training=training)
            else:
                z = layer(z)
        return self.output_layer(z)

    @staticmethod
    def calculate_physics_based_torsion(X_raw):
        """
        X_raw order:
            b, h, b_core, h_core, fc, ALfyl, Atfyt_per_s

        ALfyl and Atfyt_per_s are assumed to be compatible with N/mm mechanics:
            ALfyl      = A_l f_yl
            Atfyt/s    = A_t f_yt / s

        Output units: kN·m using N·mm to kN·m conversion = 1e-6.
        """
        b_core = tf.maximum(X_raw[:, 2], 1.0)
        h_core = tf.maximum(X_raw[:, 3], 1.0)
        fc = tf.maximum(X_raw[:, 4], 1.0)
        ALfyl = tf.maximum(X_raw[:, 5], 1e-6)
        Atfyt_per_s = tf.maximum(X_raw[:, 6], 1e-6)

        A_core = b_core * h_core
        A0 = A0_FACTOR * A_core
        ph = tf.maximum(2.0 * (b_core + h_core), 1e-6)

        theta = THETA_DEG_FIXED * np.pi / 180.0
        cot_theta = 1.0 / np.tan(theta)
        tan_theta = np.tan(theta)

        # Eq. S1
        T_s = 2.0 * A0 * Atfyt_per_s * cot_theta * 1e-6
        T_l = (2.0 * A0 * ALfyl / ph) * tan_theta * 1e-6
        T_n_phy = tf.minimum(T_s, T_l)

        # Eq. S5
        T_max = ALPHA_UPPER * T_n_phy

        # Eq. S2 cracking reference only
        T_cr = CRACKING_COEFF * tf.sqrt(fc) * (A_core ** 2 / ph) * 1e-6

        return (
            tf.reshape(T_n_phy, [-1, 1]),
            tf.reshape(T_max, [-1, 1]),
            tf.reshape(T_cr, [-1, 1]),
        )

    def physics_guided_loss(self, y_true_scaled, y_pred_scaled, X_raw, y_scaler):
        """
        Eq. S3: L_data = MSE(T_pred, T_exp)
        Eq. S4: L_physics = MSE(T_pred, T_n_phy)
        Eq. S5: L_upper = MSE(max(0, T_pred - T_max))
        Eq. S6: L_total = lambda_data L_data
                         + lambda_physics L_physics
                         + lambda_upper L_upper
        """
        T_n_phy, T_max, _ = self.calculate_physics_based_torsion(X_raw)

        y_mean = tf.constant(y_scaler.mean_, dtype=tf.float32)
        y_scale = tf.constant(y_scaler.scale_, dtype=tf.float32)

        T_n_phy_scaled = (T_n_phy - y_mean) / y_scale
        T_max_scaled = (T_max - y_mean) / y_scale

        data_loss = tf.reduce_mean(tf.square(y_pred_scaled - y_true_scaled))
        physics_loss = tf.reduce_mean(tf.square(y_pred_scaled - T_n_phy_scaled))

        upper_violation = tf.maximum(y_pred_scaled - T_max_scaled, 0.0)
        upper_loss = tf.reduce_mean(tf.square(upper_violation))

        total_loss = (
            LAMBDA_DATA * data_loss
            + self.lambda_physics * physics_loss
            + LAMBDA_UPPER * upper_loss
        )

        return total_loss, data_loss, physics_loss, upper_loss


# ============================================================
# RELIABILITY CALIBRATION
# ============================================================

def calibrate_resistance_factor(y_true, y_pred):
    eps = np.log(
        np.maximum(y_true.flatten(), 1e-9)
        / np.maximum(y_pred.flatten(), 1e-9)
    )
    mu_eps = float(np.mean(eps))
    sigma_eps = float(np.std(eps, ddof=1))

    phi_ml = float(np.exp(mu_eps - 1.645 * sigma_eps))
    gamma_ml = float(1.0 / phi_ml) if phi_ml > 0 else np.inf

    return phi_ml, gamma_ml, mu_eps, sigma_eps


# ============================================================
# TRAINING
# ============================================================

def train_model(model, Xtr_s, ytr_s, Xva_s, yva_s, Xtr_raw, Xva_raw, y_scaler):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LR,
        decay_steps=500,
        decay_rate=0.95,
        staircase=True,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    best_val_loss = np.inf
    wait = 0
    best_weights_path = OUTPUT_DIR / "best_model.weights.h5"

    history = {
        "train_loss": [],
        "val_loss": [],
        "data_loss": [],
        "physics_loss": [],
        "upper_loss": [],
        "lambda_physics": [],
        "train_rmse_scaled": [],
        "val_rmse_scaled": [],
    }

    for epoch in range(EPOCHS):
        progress = epoch / max(1, EPOCHS - 1)
        lam = LAMBDA_PHYSICS_INIT + (
            LAMBDA_PHYSICS_FINAL - LAMBDA_PHYSICS_INIT
        ) * 0.5 * (1.0 - np.cos(np.pi * progress))
        model.lambda_physics.assign(lam)

        with tf.GradientTape() as tape:
            ytr_pred = model(Xtr_s, training=True)
            loss, d_loss, p_loss, u_loss = model.physics_guided_loss(
                ytr_s, ytr_pred, Xtr_raw, y_scaler
            )

        grads = tape.gradient(loss, model.trainable_variables)
        grads_and_vars = [
            (tf.clip_by_value(g, -GRAD_CLIP, GRAD_CLIP), v)
            for g, v in zip(grads, model.trainable_variables)
            if g is not None
        ]
        optimizer.apply_gradients(grads_and_vars)

        yva_pred = model(Xva_s, training=False)
        val_loss, _, _, _ = model.physics_guided_loss(
            yva_s, yva_pred, Xva_raw, y_scaler
        )

        history["train_loss"].append(float(loss.numpy()))
        history["val_loss"].append(float(val_loss.numpy()))
        history["data_loss"].append(float(d_loss.numpy()))
        history["physics_loss"].append(float(p_loss.numpy()))
        history["upper_loss"].append(float(u_loss.numpy()))
        history["lambda_physics"].append(float(lam))

        train_rmse_scaled = float(np.sqrt(mean_squared_error(ytr_s.numpy(), ytr_pred.numpy())))
        val_rmse_scaled = float(np.sqrt(mean_squared_error(yva_s.numpy(), yva_pred.numpy())))
        history["train_rmse_scaled"].append(train_rmse_scaled)
        history["val_rmse_scaled"].append(val_rmse_scaled)

        if val_loss.numpy() < best_val_loss - 1e-8:
            best_val_loss = float(val_loss.numpy())
            wait = 0
            model.save_weights(str(best_weights_path))
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch:04d} | "
                f"Train={history['train_loss'][-1]:.4f} | "
                f"Val={history['val_loss'][-1]:.4f} | "
                f"lambda_physics={lam:.3f}"
            )

    if best_weights_path.exists():
        model.load_weights(str(best_weights_path))
        print("Loaded best model weights.")

    return history


# ============================================================
# OUTPUTS
# ============================================================

def metrics_dict(y_true, y_pred):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def save_outputs(
    model,
    history,
    y_scaler,
    Xtr_s, Xva_s, Xte_s,
    Xtr_raw, Xva_raw, Xte_raw,
    ytr_s, yva_s, yte_s,
):
    ytr_pred = y_scaler.inverse_transform(model(Xtr_s, training=False).numpy())
    yva_pred = y_scaler.inverse_transform(model(Xva_s, training=False).numpy())
    yte_pred = y_scaler.inverse_transform(model(Xte_s, training=False).numpy())

    ytr_true = y_scaler.inverse_transform(ytr_s.numpy())
    yva_true = y_scaler.inverse_transform(yva_s.numpy())
    yte_true = y_scaler.inverse_transform(yte_s.numpy())

    # ============================================================
    # ACTUAL VS PREDICTED VALUES FOR EACH DATA PARTITION
    # ============================================================
    pd.DataFrame({
        "Actual_Training_kN_m": ytr_true.flatten(),
        "Predicted_Training_kN_m": ytr_pred.flatten(),
        "Residual_Training_kN_m": (ytr_true - ytr_pred).flatten(),
    }).to_csv(OUTPUT_DIR / "training_actual_vs_predicted.csv", index=False)

    pd.DataFrame({
        "Actual_Validation_kN_m": yva_true.flatten(),
        "Predicted_Validation_kN_m": yva_pred.flatten(),
        "Residual_Validation_kN_m": (yva_true - yva_pred).flatten(),
    }).to_csv(OUTPUT_DIR / "validation_actual_vs_predicted.csv", index=False)

    pd.DataFrame({
        "Actual_Test_kN_m": yte_true.flatten(),
        "Predicted_Test_kN_m": yte_pred.flatten(),
        "Residual_Test_kN_m": (yte_true - yte_pred).flatten(),
    }).to_csv(OUTPUT_DIR / "test_actual_vs_predicted.csv", index=False)

    pd.DataFrame({
        "Dataset": (
            ["Training"] * len(ytr_true)
            + ["Validation"] * len(yva_true)
            + ["Test"] * len(yte_true)
        ),
        "Actual_kN_m": np.concatenate([
            ytr_true.flatten(),
            yva_true.flatten(),
            yte_true.flatten(),
        ]),
        "Predicted_kN_m": np.concatenate([
            ytr_pred.flatten(),
            yva_pred.flatten(),
            yte_pred.flatten(),
        ]),
        "Residual_kN_m": np.concatenate([
            (ytr_true - ytr_pred).flatten(),
            (yva_true - yva_pred).flatten(),
            (yte_true - yte_pred).flatten(),
        ]),
    }).to_csv(OUTPUT_DIR / "all_actual_vs_predicted.csv", index=False)

    Tn_tr, Tmax_tr, Tcr_tr = model.calculate_physics_based_torsion(Xtr_raw)
    Tn_va, Tmax_va, Tcr_va = model.calculate_physics_based_torsion(Xva_raw)
    Tn_te, Tmax_te, Tcr_te = model.calculate_physics_based_torsion(Xte_raw)

    Tn_tr = Tn_tr.numpy().flatten()
    Tn_va = Tn_va.numpy().flatten()
    Tn_te = Tn_te.numpy().flatten()

    Tmax_tr = Tmax_tr.numpy().flatten()
    Tmax_va = Tmax_va.numpy().flatten()
    Tmax_te = Tmax_te.numpy().flatten()

    Tcr_tr = Tcr_tr.numpy().flatten()
    Tcr_va = Tcr_va.numpy().flatten()
    Tcr_te = Tcr_te.numpy().flatten()

    all_actual = np.concatenate([ytr_true.flatten(), yva_true.flatten(), yte_true.flatten()])
    all_pred = np.concatenate([ytr_pred.flatten(), yva_pred.flatten(), yte_pred.flatten()])
    all_Tn = np.concatenate([Tn_tr, Tn_va, Tn_te])
    all_Tmax = np.concatenate([Tmax_tr, Tmax_va, Tmax_te])
    all_Tcr = np.concatenate([Tcr_tr, Tcr_va, Tcr_te])
    all_dataset = (
        ["Training"] * len(ytr_true)
        + ["Validation"] * len(yva_true)
        + ["Test"] * len(yte_true)
    )

    train_metrics = metrics_dict(ytr_true, ytr_pred)
    val_metrics = metrics_dict(yva_true, yva_pred)
    test_metrics = metrics_dict(yte_true, yte_pred)

    phi_ml, gamma_ml, mu_eps, sigma_eps = calibrate_resistance_factor(yva_true, yva_pred)

    # (1) Training history CSV
    pd.DataFrame({
        "Epoch": np.arange(1, len(history["train_loss"]) + 1),
        "Train_Loss": history["train_loss"],
        "Validation_Loss": history["val_loss"],
        "Data_Loss": history["data_loss"],
        "Physics_Loss": history["physics_loss"],
        "Upper_Loss": history["upper_loss"],
        "lambda_physics": history["lambda_physics"],
        "Train_RMSE_scaled": history["train_rmse_scaled"],
        "Validation_RMSE_scaled": history["val_rmse_scaled"],
    }).to_csv(OUTPUT_DIR / "training_history.csv", index=False)

    # (2) Loss component CSV
    pd.DataFrame({
        "Epoch": np.arange(1, len(history["data_loss"]) + 1),
        "Data_Loss": history["data_loss"],
        "Physics_Loss": history["physics_loss"],
        "Upper_Loss": history["upper_loss"],
        "Total_Training_Loss": history["train_loss"],
        "Total_Validation_Loss": history["val_loss"],
        "lambda_physics": history["lambda_physics"],
    }).to_csv(OUTPUT_DIR / "loss_components.csv", index=False)

    # (3) RMSE propagation during training CSV
    pd.DataFrame({
        "Epoch": np.arange(1, len(history["train_rmse_scaled"]) + 1),
        "Training_RMSE_scaled": history["train_rmse_scaled"],
        "Validation_RMSE_scaled": history["val_rmse_scaled"],
    }).to_csv(OUTPUT_DIR / "rmse_propagation_training.csv", index=False)

    pred_df = pd.DataFrame({
        "Dataset": all_dataset,
        "Actual_kN_m": all_actual,
        "Predicted_kN_m": all_pred,
        "Residual_kN_m": all_actual - all_pred,
        "T_n_phy_kN_m": all_Tn,
        "T_max_kN_m": all_Tmax,
        "T_cr_kN_m": all_Tcr,
        "Exceeds_Tmax": (all_pred > all_Tmax).astype(int),
    })
    pred_df.to_csv(OUTPUT_DIR / "predictions_residuals.csv", index=False)

    # (4) Actual residual analysis CSV
    # Residual defined as Actual - Predicted and grouped by actual torsional capacity.
    actual_residual_df = pred_df.copy()
    actual_residual_df["Abs_Residual_kN_m"] = actual_residual_df["Residual_kN_m"].abs()
    actual_residual_df["Residual_Percent_of_Actual"] = (
        100.0 * actual_residual_df["Residual_kN_m"] /
        np.maximum(np.abs(actual_residual_df["Actual_kN_m"]), 1e-9)
    )
    actual_residual_df["Actual_Bin"] = pd.qcut(
        actual_residual_df["Actual_kN_m"],
        q=10,
        duplicates="drop"
    )
    actual_residual_df.to_csv(OUTPUT_DIR / "actual_residual_analysis.csv", index=False)

    actual_residual_summary = actual_residual_df.groupby(
        ["Dataset", "Actual_Bin"], observed=False
    ).agg(
        n=("Residual_kN_m", "size"),
        Actual_Mean_kN_m=("Actual_kN_m", "mean"),
        Predicted_Mean_kN_m=("Predicted_kN_m", "mean"),
        Residual_Mean_kN_m=("Residual_kN_m", "mean"),
        Residual_Std_kN_m=("Residual_kN_m", "std"),
        MAE_kN_m=("Abs_Residual_kN_m", "mean"),
        Mean_Residual_Percent=("Residual_Percent_of_Actual", "mean"),
    ).reset_index()
    actual_residual_summary.to_csv(OUTPUT_DIR / "actual_residual_summary.csv", index=False)

    # (5) Predicted residual analysis CSV
    # Residual defined as Actual - Predicted and grouped by predicted torsional capacity.
    predicted_residual_df = pred_df.copy()
    predicted_residual_df["Abs_Residual_kN_m"] = predicted_residual_df["Residual_kN_m"].abs()
    predicted_residual_df["Residual_Percent_of_Predicted"] = (
        100.0 * predicted_residual_df["Residual_kN_m"] /
        np.maximum(np.abs(predicted_residual_df["Predicted_kN_m"]), 1e-9)
    )
    predicted_residual_df["Predicted_Bin"] = pd.qcut(
        predicted_residual_df["Predicted_kN_m"],
        q=10,
        duplicates="drop"
    )
    predicted_residual_df.to_csv(OUTPUT_DIR / "predicted_residual_analysis.csv", index=False)

    predicted_residual_summary = predicted_residual_df.groupby(
        ["Dataset", "Predicted_Bin"], observed=False
    ).agg(
        n=("Residual_kN_m", "size"),
        Actual_Mean_kN_m=("Actual_kN_m", "mean"),
        Predicted_Mean_kN_m=("Predicted_kN_m", "mean"),
        Residual_Mean_kN_m=("Residual_kN_m", "mean"),
        Residual_Std_kN_m=("Residual_kN_m", "std"),
        MAE_kN_m=("Abs_Residual_kN_m", "mean"),
        Mean_Residual_Percent=("Residual_Percent_of_Predicted", "mean"),
    ).reset_index()
    predicted_residual_summary.to_csv(OUTPUT_DIR / "predicted_residual_summary.csv", index=False)

    pd.DataFrame({
        "Dataset": ["Training", "Validation", "Test"],
        "MSE": [train_metrics["MSE"], val_metrics["MSE"], test_metrics["MSE"]],
        "RMSE": [train_metrics["RMSE"], val_metrics["RMSE"], test_metrics["RMSE"]],
        "MAE": [train_metrics["MAE"], val_metrics["MAE"], test_metrics["MAE"]],
        "R2": [train_metrics["R2"], val_metrics["R2"], test_metrics["R2"]],
    }).to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    exceedance_summary = pred_df.groupby("Dataset").agg(
        n=("Exceeds_Tmax", "size"),
        Violations=("Exceeds_Tmax", "sum"),
    ).reset_index()
    exceedance_summary["Violation_rate_%"] = (
        100.0 * exceedance_summary["Violations"] / exceedance_summary["n"]
    )
    exceedance_summary.to_csv(OUTPUT_DIR / "exceedance_summary.csv", index=False)

    corr_df = pd.DataFrame({
        "Pair": [
            "Predicted vs T_n_phy",
            "Predicted vs T_max",
            "Actual vs T_n_phy",
            "Actual vs T_max",
        ],
        "Pearson": [
            np.corrcoef(all_pred, all_Tn)[0, 1],
            np.corrcoef(all_pred, all_Tmax)[0, 1],
            np.corrcoef(all_actual, all_Tn)[0, 1],
            np.corrcoef(all_actual, all_Tmax)[0, 1],
        ],
    })
    corr_df.to_csv(OUTPUT_DIR / "anchors_correlations.csv", index=False)

    pd.DataFrame({
        "mu_epsilon": [mu_eps],
        "sigma_epsilon": [sigma_eps],
        "phi_ML": [phi_ml],
        "gamma_ML": [gamma_ml],
        "calibration_dataset": ["Validation"],
    }).to_csv(OUTPUT_DIR / "reliability_calibration.csv", index=False)

    pd.DataFrame({
        "Equation": ["S1", "S2", "S3", "S4", "S5", "S6", "S7"],
        "Description": [
            "T_n_phy = min[2A0(Atfyt/s)cot(theta), 2A0(ALfyl/ph)tan(theta)]",
            "T_cr = 0.33 sqrt(fc) A_core^2 / ph",
            "L_data = MSE(T_pred, T_exp)",
            "L_physics = MSE(T_pred, T_n_phy)",
            "L_upper = MSE(max(0, T_pred - alpha T_n_phy))",
            "L_total = lambda_data L_data + lambda_physics L_physics + lambda_upper L_upper",
            "phi_ML = exp(mu_epsilon - 1.645 sigma_epsilon)",
        ],
    }).to_csv(OUTPUT_DIR / "manuscript_equations_summary.csv", index=False)

    # ============================================================
    # REQUESTED FIGURES ONLY — model/training logic is unchanged
    # ============================================================

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    split_data = {
        "Training": {
            "actual": ytr_true.flatten(),
            "pred": ytr_pred.flatten(),
            "res": (ytr_true - ytr_pred).flatten(),
            "marker": "o",
        },
        "Validation": {
            "actual": yva_true.flatten(),
            "pred": yva_pred.flatten(),
            "res": (yva_true - yva_pred).flatten(),
            "marker": "s",
        },
        "Test": {
            "actual": yte_true.flatten(),
            "pred": yte_pred.flatten(),
            "res": (yte_true - yte_pred).flatten(),
            "marker": "^",
        },
    }

    # ------------------------------------------------------------
    # Loss component plot
    # ------------------------------------------------------------
    plt.figure(figsize=(9, 6))
    plt.plot(epochs, history["data_loss"], linewidth=2.0, label="Data loss")
    plt.plot(epochs, history["physics_loss"], linewidth=2.0, label="Physics loss")
    plt.plot(epochs, history["upper_loss"], linewidth=2.0, label="Upper-bound loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Component Evolution")
    plt.grid(True, alpha=0.30)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_component_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------
    # (c) Training history
    # ------------------------------------------------------------
    plt.figure(figsize=(9, 6))
    plt.plot(epochs, history["train_loss"], linewidth=2.0, label="Training loss")
    plt.plot(epochs, history["val_loss"], linewidth=2.0, label="Validation loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Total loss")
    plt.title("(c) PGNN Training History")
    plt.grid(True, alpha=0.30)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "c_training_history.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------
    # (d) RMSE propagation during training
    # ------------------------------------------------------------
    plt.figure(figsize=(9, 6))
    plt.plot(epochs, history["train_rmse_scaled"], linewidth=2.0, label="Training RMSE")
    plt.plot(epochs, history["val_rmse_scaled"], linewidth=2.0, label="Validation RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("Scaled RMSE")
    plt.title("(d) RMSE Propagation During Training")
    plt.grid(True, alpha=0.30)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "d_rmse_propagation_training.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------
    # (e) Actual residual analysis for training, validation, and test
    # Residual = Actual - Predicted
    # ------------------------------------------------------------
    plt.figure(figsize=(9, 6))
    for name, data in split_data.items():
        plt.scatter(
            data["actual"],
            data["res"],
            s=46,
            alpha=0.75,
            marker=data["marker"],
            edgecolors="black",
            linewidths=0.45,
            label=name,
        )
    plt.axhline(0.0, linestyle="--", linewidth=1.5)
    plt.xlabel("Actual torsional capacity, $T_{exp}$ (kN·m)")
    plt.ylabel("Residual, $T_{exp}-T_{pred}$ (kN·m)")
    plt.title("(e) Actual Residual Analysis")
    plt.grid(True, alpha=0.30)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "e_actual_residual_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------
    # (f) Predicted residual analysis for training, validation, and test
    # Residual = Actual - Predicted
    # ------------------------------------------------------------
    plt.figure(figsize=(9, 6))
    for name, data in split_data.items():
        plt.scatter(
            data["pred"],
            data["res"],
            s=46,
            alpha=0.75,
            marker=data["marker"],
            edgecolors="black",
            linewidths=0.45,
            label=name,
        )
    plt.axhline(0.0, linestyle="--", linewidth=1.5)
    plt.xlabel("Predicted torsional capacity, $T_{pred}$ (kN·m)")
    plt.ylabel("Residual, $T_{exp}-T_{pred}$ (kN·m)")
    plt.title("(f) Predicted Residual Analysis")
    plt.grid(True, alpha=0.30)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "f_predicted_residual_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------
    # Combined manuscript-style figure with requested panels
    # ------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(epochs, history["data_loss"], linewidth=2.0, label="Data")
    axes[0, 0].plot(epochs, history["physics_loss"], linewidth=2.0, label="Physics")
    axes[0, 0].plot(epochs, history["upper_loss"], linewidth=2.0, label="Upper")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Loss Component Evolution")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(frameon=True)
    axes[0, 0].grid(True, alpha=0.30)

    axes[0, 1].plot(epochs, history["train_loss"], linewidth=2.0, label="Training")
    axes[0, 1].plot(epochs, history["val_loss"], linewidth=2.0, label="Validation")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("(c) Training History")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Total Loss")
    axes[0, 1].legend(frameon=True)
    axes[0, 1].grid(True, alpha=0.30)

    axes[0, 2].plot(epochs, history["train_rmse_scaled"], linewidth=2.0, label="Training")
    axes[0, 2].plot(epochs, history["val_rmse_scaled"], linewidth=2.0, label="Validation")
    axes[0, 2].set_title("(d) RMSE Propagation")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Scaled RMSE")
    axes[0, 2].legend(frameon=True)
    axes[0, 2].grid(True, alpha=0.30)

    for name, data in split_data.items():
        axes[1, 0].scatter(
            data["actual"], data["res"],
            s=42, alpha=0.75, marker=data["marker"],
            edgecolors="black", linewidths=0.45, label=name
        )
    axes[1, 0].axhline(0.0, linestyle="--", linewidth=1.5)
    axes[1, 0].set_title("(e) Actual Residual Analysis")
    axes[1, 0].set_xlabel("Actual $T_{exp}$ (kN·m)")
    axes[1, 0].set_ylabel("Residual (kN·m)")
    axes[1, 0].legend(frameon=True)
    axes[1, 0].grid(True, alpha=0.30)

    for name, data in split_data.items():
        axes[1, 1].scatter(
            data["pred"], data["res"],
            s=42, alpha=0.75, marker=data["marker"],
            edgecolors="black", linewidths=0.45, label=name
        )
    axes[1, 1].axhline(0.0, linestyle="--", linewidth=1.5)
    axes[1, 1].set_title("(f) Predicted Residual Analysis")
    axes[1, 1].set_xlabel("Predicted $T_{pred}$ (kN·m)")
    axes[1, 1].set_ylabel("Residual (kN·m)")
    axes[1, 1].legend(frameon=True)
    axes[1, 1].grid(True, alpha=0.30)

    axes[1, 2].hist(
        [split_data["Training"]["res"], split_data["Validation"]["res"], split_data["Test"]["res"]],
        bins=28,
        alpha=0.70,
        label=["Training", "Validation", "Test"],
        density=True,
    )
    axes[1, 2].axvline(0.0, linestyle="--", linewidth=1.5)
    axes[1, 2].set_title("Residual Distribution")
    axes[1, 2].set_xlabel("Residual (kN·m)")
    axes[1, 2].set_ylabel("Density")
    axes[1, 2].legend(frameon=True)
    axes[1, 2].grid(True, alpha=0.30)

    plt.suptitle(f"PGNN Training and Residual Diagnostics | theta={THETA_DEG_FIXED}°, alpha={ALPHA_UPPER}", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "PGNN_requested_diagnostic_panels.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nFinal performance:")
    for name, md in zip(["Training", "Validation", "Test"], [train_metrics, val_metrics, test_metrics]):
        print(f"{name:10s}: R²={md['R2']:.4f}, RMSE={md['RMSE']:.2f}, MAE={md['MAE']:.2f}")

    print("\nReliability calibration:")
    print(f"phi_ML={phi_ml:.4f}, gamma_ML={gamma_ml:.4f}, mu_eps={mu_eps:.4f}, sigma_eps={sigma_eps:.4f}")

    print("\nExceedance summary:")
    print(exceedance_summary.to_string(index=False))

    print("\nAdditional CSV files saved:")
    print("  1. loss_components.csv")
    print("  2. training_history.csv")
    print("  3. rmse_propagation_training.csv")
    print("  4. actual_residual_analysis.csv")
    print("  5. actual_residual_summary.csv")
    print("  6. predicted_residual_analysis.csv")
    print("  7. predicted_residual_summary.csv")
    print("  8. training_actual_vs_predicted.csv")
    print("  9. validation_actual_vs_predicted.csv")
    print(" 10. test_actual_vs_predicted.csv")
    print(" 11. all_actual_vs_predicted.csv")
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


# ============================================================
# MAIN
# ============================================================

def main():
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        csv_path = Path(r"C:\Users\owner\Desktop\PGNN\Dataset.csv")

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = load_dataset(csv_path)
    print(f"Loaded dataset: {csv_path}")
    print(f"Samples after cleaning: {len(df)}")
    print("Columns:", list(df.columns))

    X_raw = df[["b", "h", "b_core", "h_core", "fc", "ALfyl", "Atfyt_per_s"]].values.astype(np.float32)
    y = df["Ts"].values.reshape(-1, 1).astype(np.float32)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_raw, y, test_size=TEST_SIZE, random_state=SEED
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=VAL_SIZE_OF_REMAINING, random_state=SEED
    )

    X_train, y_train = winsorize_train_only(X_train, y_train)

    # Enhanced features for the neural network only.
    # Raw features are still used for physics equations.
    Xtr_enh = make_enhanced_features(X_train)
    Xva_enh = make_enhanced_features(X_val)
    Xte_enh = make_enhanced_features(X_test)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    Xtr_s = x_scaler.fit_transform(Xtr_enh).astype(np.float32)
    Xva_s = x_scaler.transform(Xva_enh).astype(np.float32)
    Xte_s = x_scaler.transform(Xte_enh).astype(np.float32)

    ytr_s = y_scaler.fit_transform(y_train).astype(np.float32)
    yva_s = y_scaler.transform(y_val).astype(np.float32)
    yte_s = y_scaler.transform(y_test).astype(np.float32)

    Xtr_s_tf = tf.convert_to_tensor(Xtr_s, dtype=tf.float32)
    Xva_s_tf = tf.convert_to_tensor(Xva_s, dtype=tf.float32)
    Xte_s_tf = tf.convert_to_tensor(Xte_s, dtype=tf.float32)

    ytr_s_tf = tf.convert_to_tensor(ytr_s, dtype=tf.float32)
    yva_s_tf = tf.convert_to_tensor(yva_s, dtype=tf.float32)
    yte_s_tf = tf.convert_to_tensor(yte_s, dtype=tf.float32)

    Xtr_raw_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
    Xva_raw_tf = tf.convert_to_tensor(X_val, dtype=tf.float32)
    Xte_raw_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)

    print("\nData split:")
    print(f"  Training:   {len(X_train)}")
    print(f"  Validation: {len(X_val)}")
    print(f"  Test:       {len(X_test)}")

    model = PhysicsGuidedTorsionModel()
    history = train_model(
        model,
        Xtr_s_tf, ytr_s_tf,
        Xva_s_tf, yva_s_tf,
        Xtr_raw_tf, Xva_raw_tf,
        y_scaler,
    )

    save_outputs(
        model,
        history,
        y_scaler,
        Xtr_s_tf, Xva_s_tf, Xte_s_tf,
        Xtr_raw_tf, Xva_raw_tf, Xte_raw_tf,
        ytr_s_tf, yva_s_tf, yte_s_tf,
    )


if __name__ == "__main__":
    main()
