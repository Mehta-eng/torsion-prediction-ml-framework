import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------
# Plotting style
# ---------------------------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# ---------------------------
# Feature name handling
# ---------------------------
# 1) Physical names you want on plots
PHYSICAL_NAMES = {
    "b": "b (mm)",
    "h": "h (mm)",
    "bp": "b′ (mm)",
    "hp": "h′ (mm)",
    "fc": "f'c (MPa)",
    "ALfyL": "ALfyL (kN)",
    "Atfyt/s": "Atfyt/s (N/mm)",
}

# 2) If CSV uses X1..X7,Y8 map them to physical columns (dataset aliases)
ALIAS_FROM_X = {
    "X1": "b",
    "X2": "h",
    "X3": "bp",
    "X4": "hp",
    "X5": "fc",
    "X6": "ALfyL",
    "X7": "Atfyt/s",
    "Y8": "Ts",  # target
}

# 3) Some users might use different literal column names for bp/hp
CANONICALIZE = {
    "b'": "bp",
    "h'": "hp",
    "b_prime": "bp",
    "h_prime": "hp",
    "fc'": "fc",
    "f'c": "fc",
    "f'c (MPa)": "fc",
    "ALfyL (kN)": "ALfyL",
    "At/s": "Atfyt/s",
    "At_fyt_s": "Atfyt/s",
}

POSSIBLE_TARGETS = [
    'Tr', 'Torsion', 'Torsional_Resistance', 'Torsion_Resistance',
    'Torsional_Capacity', 'Torsion_Capacity', 'Target', 'y', 'Y8', 'Ts'
]

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Unify column names to a canonical set {b,h,bp,hp,fc,ALfyL,Atfyt/s,Ts}."""
    cols = list(df.columns)

    # First map any X1..X7,Y8 to physical short names
    rename_map = {}
    for c in cols:
        if c in ALIAS_FROM_X:
            rename_map[c] = ALIAS_FROM_X[c]
    df = df.rename(columns=rename_map)

    # Canonicalize alternative spellings
    rename_map = {}
    for c in df.columns:
        if c in CANONICALIZE:
            rename_map[c] = CANONICALIZE[c]
    df = df.rename(columns=rename_map)

    return df

def physical_feature_labels(feature_keys):
    """Convert canonical feature keys to plot labels with units."""
    return [PHYSICAL_NAMES.get(k, k) for k in feature_keys]

# ---------------------------
# Data load & preprocess
# ---------------------------
def load_and_preprocess_data(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    data = pd.read_csv(dataset_path)
    print("Raw columns:", data.columns.tolist(), "| shape:", data.shape)

    # Standardize column names (X1->b, b'->bp, etc.)
    data = standardize_columns(data)
    print("Standardized columns:", data.columns.tolist())

    # Identify target
    target_column = None
    for col in POSSIBLE_TARGETS:
        if col in data.columns:
            target_column = col
            break
    if target_column is None:
        target_column = data.columns[-1]
        print(f"Using last column as target: '{target_column}'")
    else:
        print(f"Using detected target column: '{target_column}'")

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column].astype(float).values

    # Keep only the 7 expected features in canonical order if present
    expected_order = ["b", "h", "bp", "hp", "fc", "ALfyL", "Atfyt/s"]
    # Intersect keeping order
    final_cols = [c for c in expected_order if c in X.columns]
    # If user provided extras, append them after expected ones (still supported)
    extras = [c for c in X.columns if c not in final_cols]
    feature_keys = final_cols + extras
    X = X[feature_keys].astype(float)

    # Split (note: we scale but XGBoost often works fine without; we'll use scaled for SHAP consistency)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)
    joblib.dump(scaler, 'scaler.joblib')

    # Feature names for plots (physical labels)
    feature_plot_labels = physical_feature_labels(feature_keys)

    return {
        'feature_keys': feature_keys,               # canonical keys used for arrays
        'feature_plot_labels': feature_plot_labels, # pretty labels for plots
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'X_train_df': X_train_df, 'X_test_df': X_test_df,
        'target_name': target_column
    }

# ---------------------------
# Model train/load
# ---------------------------
def train_xgboost_model(data_dict):
    print("Training XGBoost model...")
    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    model.fit(data_dict['X_train'], data_dict['y_train'])
    y_pred = model.predict(data_dict['X_test'])
    mse = mean_squared_error(data_dict['y_test'], y_pred)
    r2 = r2_score(data_dict['y_test'], y_pred)
    print(f"Model trained | Test MSE: {mse:.4f} | R²: {r2:.4f}")
    model.save_model('xgb_model.json')
    print("Saved model -> xgb_model.json")
    return model

def load_data_and_model(dataset_path):
    data = load_and_preprocess_data(dataset_path)

    model_path = 'xgb_model.json'
    model = None
    if os.path.exists(model_path):
        try:
            # Prefer sklearn API if you trained one; try to load Booster and wrap if needed
            booster = xgb.Booster()
            booster.load_model(model_path)
            print("Loaded Booster from xgb_model.json")
            # Wrap Booster to sklearn-like model using XGBRegressor with .load_model
            model = XGBRegressor()
            model.load_model(model_path)
            print("Wrapped Booster into XGBRegressor for SHAP")
        except Exception as e:
            print("Failed to load existing model as Booster:", e)
    if model is None or not hasattr(model, "predict"):
        model = train_xgboost_model(data)

    # Build DataFrame version of test set with pretty labels for SHAP visuals
    X_test_df_pretty = pd.DataFrame(
        data['X_test'],
        columns=data['feature_plot_labels']
    )
    return model, data, X_test_df_prety_safe(X_test_df_pretty), data['target_name']

def X_test_df_prety_safe(df):
    """Some SHAP versions are picky—ensure no duplicate labels and all are strings."""
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    # De-duplicate just in case
    if len(set(df.columns)) != len(df.columns):
        new_cols = []
        seen = {}
        for c in df.columns:
            if c not in seen:
                seen[c] = 1
                new_cols.append(c)
            else:
                seen[c] += 1
                new_cols.append(f"{c} ({seen[c]})")
        df.columns = new_cols
    return df

# ---------------------------
# SHAP analysis
# ---------------------------
def perform_shap_analysis(dataset_path):
    print("Loading data and model...")
    model, data, X_test_pretty, target_name = load_data_and_model(dataset_path)

    feature_plot_labels = list(X_test_pretty.columns)
    print(f"Model expects {getattr(model, 'n_features_in_', 'unknown')} features; test has {X_test_pretty.shape[1]}.")

    # TreeExplainer
    explainer = shap.TreeExplainer(model)
    print("Calculating SHAP values...")
    try:
        shap_values = explainer.shap_values(X_test_pretty)
    except Exception:
        shap_values = explainer.shap_values(X_test_pretty.values)

    # 1) Summary plot (beeswarm)
    print("Summary plot...")
    plt.figure(figsize=(12, 9))
    shap.summary_plot(shap_values, X_test_pretty, feature_names=feature_plot_labels, show=False)
    plt.title("SHAP Feature Importance Summary", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_feature_importance.png', bbox_inches='tight')
    plt.close()

    # 2) Bar plot of mean |SHAP|
    print("Bar plot...")
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test_pretty, feature_names=feature_plot_labels,
                      plot_type="bar", show=False)
    plt.title("Mean Absolute SHAP Values", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_mean_absolute.png', bbox_inches='tight')
    plt.close()

    # 3) Dependence plots for each feature
    print("Dependence plots...")
    mean_abs = np.abs(shap_values).mean(axis=0)
    for i, f in enumerate(feature_plot_labels):
        # Color by the most influential *other* feature
        other = mean_abs.copy()
        other[i] = -np.inf
        j = int(np.argmax(other))
        color_by = feature_plot_labels[j]

        plt.figure(figsize=(10, 7))
        shap.dependence_plot(
            f, shap_values, X_test_pretty, interaction_index=color_by, show=False
        )
        ax = plt.gca()
        for col in getattr(ax, "collections", []):
            try:
                col.set_sizes([60])
            except Exception:
                pass
        plt.title(f"SHAP Dependence: {f}\n(colored by {color_by})", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"shap_dependence_{safe_name(f)}.png", bbox_inches='tight')
        plt.close()

    # 4) Force plots for samples (first, middle, last)
    print("Force plots...")
    base_val = explainer.expected_value
    idxs = [0, len(X_test_pretty)//2, len(X_test_pretty)-1]
    for idx in idxs:
        plt.figure(figsize=(14, 6))
        shap.force_plot(
            base_val, shap_values[idx, :], X_test_pretty.iloc[idx, :],
            feature_names=feature_plot_labels, matplotlib=True, show=False
        )
        plt.title(f"SHAP Force Plot — Sample {idx} (Actual: {data['y_test'][idx]:.2f} {target_name})",
                  fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"shap_force_{idx}.png", bbox_inches='tight')
        plt.close()

    # 5) Waterfall plot for sample 0
    print("Waterfall plot...")
    sample_idx = 0
    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=base_val,
        data=X_test_pretty.iloc[sample_idx],
        feature_names=feature_plot_labels
    )
    plt.figure(figsize=(12, 9))
    shap.waterfall_plot(explanation, show=False, max_display=min(12, len(feature_plot_labels)))
    plt.title(f"SHAP Waterfall — Sample {sample_idx} (Actual: {data['y_test'][sample_idx]:.2f} {target_name})",
              fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("shap_waterfall.png", bbox_inches='tight')
    plt.close()

    # 6) Heatmap
    print("Heatmap...")
    try:
        explanation_all = shap.Explanation(
            values=shap_values,
            base_values=base_val,
            data=X_test_pretty,
            feature_names=feature_plot_labels
        )
        order = np.argsort(shap_values.sum(1))
        plt.figure(figsize=(12, 9))
        shap.plots.heatmap(explanation_all, show=False, instance_order=order)
        plt.title("SHAP Values Heatmap", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig("shap_heatmap.png", bbox_inches='tight')
        plt.close()
    except Exception as e:
        print("Heatmap failed:", e)

    print("SHAP analysis complete. Plots saved to current folder.")
    return explainer, shap_values, target_name, X_test_pretty

def safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s)

# ---------------------------
# Interpretation (optional console)
# ---------------------------
def interpret_shap_results(explainer, shap_values, X_test_pretty, target_name):
    print("\n=== SHAP ANALYSIS INTERPRETATION ===")
    feature_names = list(X_test_pretty.columns)
    mean_abs = np.abs(shap_values).mean(axis=0)
    imp = pd.DataFrame({"Feature": feature_names, "Importance": mean_abs}).sort_values(
        "Importance", ascending=False
    )
    print("\nTop 5 Features:")
    for _, r in imp.head(5).iterrows():
        print(f"  {r['Feature']}: {r['Importance']:.4f}")

    print("\nPhysical intuition check:")
    print("  • Expect positive influence of ALfyL and Atfyt/s on torsional capacity.")
    print("  • Size terms (b, h, bp, hp) typically increase capacity; fc helps but with diminishing returns.")

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python shap_rc_torsion.py <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    explainer, shap_values, target_name, X_test_pretty = perform_shap_analysis(dataset_path)
    interpret_shap_results(explainer, shap_values, X_test_pretty, target_name)
