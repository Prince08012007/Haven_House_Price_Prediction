"""Inference engine and model management for House Price Prediction ML.
Provides single-sample and batch prediction, feature engineering, and model loading/caching.
"""

from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_BUNDLE_PATH = OUT_DIR / "model_bundle.joblib"
RANDOM_STATE = 42

DEFAULT_BLEND_WEIGHTS = {
    "CatBoost": 0.618763,
    "XGBoost": 0.381237,
    "GradientBoosting": 0.0,
}


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create consistent domain features for both training and test rows."""
    df = frame.copy()

    def s(name):
        if name in df.columns:
            return df[name]
        return pd.Series(np.nan, index=df.index)

    # Area and room features
    df["TotalSF"] = s("TotalBsmtSF").fillna(0) + s("1stFlrSF").fillna(0) + s("2ndFlrSF").fillna(0)
    df["TotalPorchSF"] = sum(
        s(c).fillna(0) for c in ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"]
    )
    df["TotalBathrooms"] = (
        s("FullBath").fillna(0)
        + 0.5 * s("HalfBath").fillna(0)
        + s("BsmtFullBath").fillna(0)
        + 0.5 * s("BsmtHalfBath").fillna(0)
    )
    df["TotalRoomsPlusBedrooms"] = s("TotRmsAbvGrd").fillna(0) + s("BedroomAbvGr").fillna(0)

    # Age and renovation features; clip negative values
    df["HouseAgeAtSale"] = (s("YrSold") - s("YearBuilt")).clip(lower=0)
    df["YearsSinceRemodel"] = (s("YrSold") - s("YearRemodAdd")).clip(lower=0)
    df["GarageAgeAtSale"] = (s("YrSold") - s("GarageYrBlt")).clip(lower=0)

    # Quality interactions and presence flags
    df["Quality_x_LivingArea"] = s("OverallQual").fillna(0) * s("GrLivArea").fillna(0)
    df["Quality_x_TotalSF"] = s("OverallQual").fillna(0) * df["TotalSF"]
    df["HasGarage"] = (s("GarageArea").fillna(0) > 0).astype(int)
    df["HasBasement"] = (s("TotalBsmtSF").fillna(0) > 0).astype(int)
    df["HasPool"] = (s("PoolArea").fillna(0) > 0).astype(int)
    df["HasFireplace"] = (s("Fireplaces").fillna(0) > 0).astype(int)
    df["Has2ndFloor"] = (s("2ndFlrSF").fillna(0) > 0).astype(int)

    # Ratios
    df["LivingAreaPerRoom"] = s("GrLivArea") / s("TotRmsAbvGrd").replace(0, np.nan)
    df["GarageAreaPerCar"] = s("GarageArea") / s("GarageCars").replace(0, np.nan)
    return df.replace([np.inf, -np.inf], np.nan)


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])


def find_data_file(filename: str) -> Path:
    candidates = [
        DATA_DIR / filename,
        ROOT.parent / "house_prices_advanced_regression" / "data" / filename,
        ROOT.parent / "data" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return DATA_DIR / filename


def load_training_data() -> pd.DataFrame:
    path = find_data_file("train.csv")
    if not path.exists():
        raise FileNotFoundError(f"Training data not found at {path}")
    return pd.read_csv(path)


def compute_baseline_defaults(train_df: pd.DataFrame) -> dict:
    """Compute default median/mode values for all raw columns so single predictions work with partial inputs."""
    defaults = {}
    cols = [c for c in train_df.columns if c not in ["Id", "SalePrice"]]
    for col in cols:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            defaults[col] = float(train_df[col].median(skipna=True))
        else:
            mode_val = train_df[col].mode(dropna=True)
            defaults[col] = mode_val.iloc[0] if len(mode_val) > 0 else "None"
    return defaults


def train_and_bundle_models(progress_callback=None) -> dict:
    """Train the V3 models (CatBoost, XGBoost, GradientBoosting) and save to joblib bundle."""
    if progress_callback:
        progress_callback("Loading training data...", 0.1)

    train = load_training_data()
    y = np.log1p(train["SalePrice"].astype(float)).to_numpy()
    raw_X = train.drop(columns=["Id", "SalePrice"])

    baseline_defaults = compute_baseline_defaults(train)
    X = add_features(raw_X)

    if progress_callback:
        progress_callback("Fitting preprocessing transformer...", 0.25)

    preprocessor = make_preprocessor(X)
    X_trans = preprocessor.fit_transform(X)

    models = {}

    # Gradient Boosting (scikit-learn)
    if progress_callback:
        progress_callback("Training Gradient Boosting...", 0.45)
    gb = GradientBoostingRegressor(
        n_estimators=350,
        learning_rate=0.04,
        max_depth=2,
        max_features=0.8,
        loss="huber",
        random_state=RANDOM_STATE,
    )
    gb.fit(X_trans, y)
    models["GradientBoosting"] = gb

    # XGBoost
    try:
        from xgboost import XGBRegressor
        if progress_callback:
            progress_callback("Training XGBoost Regressor...", 0.65)
        xgb = XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=2,
            subsample=0.75,
            colsample_bytree=0.8,
            reg_alpha=0.02,
            reg_lambda=1.5,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        xgb.fit(X_trans, y)
        models["XGBoost"] = xgb
    except ImportError:
        pass

    # CatBoost
    try:
        from catboost import CatBoostRegressor
        if progress_callback:
            progress_callback("Training CatBoost Regressor...", 0.85)
        cb = CatBoostRegressor(
            iterations=500,
            learning_rate=0.04,
            depth=4,
            loss_function="RMSE",
            l2_leaf_reg=4,
            verbose=False,
            random_seed=RANDOM_STATE,
            allow_writing_files=False,
        )
        cb.fit(X_trans, y)
        models["CatBoost"] = cb
    except ImportError:
        pass

    bundle = {
        "preprocessor": preprocessor,
        "models": models,
        "weights": DEFAULT_BLEND_WEIGHTS,
        "feature_columns": X.columns.tolist(),
        "baseline_defaults": baseline_defaults,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        joblib.dump(bundle, MODEL_BUNDLE_PATH, compress=3)
    except Exception as exc:
        print(f"Notice: Could not write bundle to {MODEL_BUNDLE_PATH} ({exc}). Kept in memory.")

    if progress_callback:
        progress_callback("Models ready!", 1.0)

    return bundle


_CACHED_BUNDLE = None


def get_model_bundle(force_retrain=False, progress_callback=None) -> dict:
    """Retrieve cached model bundle or load from disk / train if necessary."""
    global _CACHED_BUNDLE
    if _CACHED_BUNDLE is not None and not force_retrain:
        return _CACHED_BUNDLE

    if MODEL_BUNDLE_PATH.exists() and not force_retrain:
        try:
            _CACHED_BUNDLE = joblib.load(MODEL_BUNDLE_PATH)
            return _CACHED_BUNDLE
        except Exception:
            pass

    _CACHED_BUNDLE = train_and_bundle_models(progress_callback=progress_callback)
    return _CACHED_BUNDLE


def predict_single(user_inputs: dict, model_name: str = "Ensemble Blend") -> dict:
    """Make prediction for a single house given partial or complete attributes."""
    bundle = get_model_bundle()
    defaults = bundle["baseline_defaults"].copy()

    # Update defaults with user-supplied values
    for k, v in user_inputs.items():
        if v is not None and v != "":
            defaults[k] = v

    # Convert to single-row DataFrame
    raw_df = pd.DataFrame([defaults])
    feat_df = add_features(raw_df)

    # Reindex columns to match training set
    feat_df = feat_df.reindex(columns=bundle["feature_columns"])
    X_trans = bundle["preprocessor"].transform(feat_df)

    predictions_log = {}
    predictions_price = {}

    for name, model in bundle["models"].items():
        log_pred = float(model.predict(X_trans)[0])
        predictions_log[name] = log_pred
        predictions_price[name] = float(np.maximum(np.expm1(log_pred), 1.0))

    # Ensemble Weighted Blend calculation
    weights = bundle["weights"]
    available_models = [m for m in bundle["models"] if m in weights and weights[m] > 0]
    total_w = sum(weights[m] for m in available_models)

    if total_w > 0:
        norm_weights = {m: weights[m] / total_w for m in available_models}
        blend_log = sum(predictions_log[m] * norm_weights[m] for m in available_models)
        blend_price = float(np.maximum(np.expm1(blend_log), 1.0))
    else:
        # Fallback to mean of all available models
        blend_log = float(np.mean(list(predictions_log.values())))
        blend_price = float(np.maximum(np.expm1(blend_log), 1.0))

    predictions_price["Ensemble Blend"] = blend_price

    chosen_price = predictions_price.get(model_name, blend_price)

    # Calculate reasonable confidence interval based on V2/V3 test RMSE (~0.124 log RMSE)
    # exp(log_pred - 0.124) to exp(log_pred + 0.124)
    low_bound = float(np.maximum(np.expm1(np.log1p(chosen_price) - 0.124), 1.0))
    high_bound = float(np.maximum(np.expm1(np.log1p(chosen_price) + 0.124), 1.0))

    return {
        "predicted_price": round(chosen_price, 2),
        "price_low": round(low_bound, 2),
        "price_high": round(high_bound, 2),
        "model_breakdown": {k: round(v, 2) for k, v in predictions_price.items()},
        "weights_used": bundle["weights"],
        "engineered_summary": {
            "Total Living Area (GrLivArea)": f"{int(defaults.get('GrLivArea', 0)):,} sq ft",
            "Total Basement (TotalBsmtSF)": f"{int(defaults.get('TotalBsmtSF', 0)):,} sq ft",
            "Total Calculated SF": f"{int(feat_df['TotalSF'].iloc[0]):,} sq ft",
            "Total Bathrooms": f"{feat_df['TotalBathrooms'].iloc[0]:.1f}",
            "House Age at Sale": f"{int(feat_df['HouseAgeAtSale'].iloc[0])} years",
            "Quality × Living Area": f"{int(feat_df['Quality_x_LivingArea'].iloc[0]):,}",
        },
    }


def predict_batch(df: pd.DataFrame, model_name: str = "Ensemble Blend") -> pd.DataFrame:
    """Generate predictions for a test dataset DataFrame."""
    bundle = get_model_bundle()
    test_ids = df["Id"].copy() if "Id" in df.columns else pd.Series(range(1, len(df) + 1))

    raw_X = df.drop(columns=["Id", "SalePrice"], errors="ignore")
    feat_df = add_features(raw_X)
    feat_df = feat_df.reindex(columns=bundle["feature_columns"])

    X_trans = bundle["preprocessor"].transform(feat_df)

    if model_name in bundle["models"]:
        log_preds = bundle["models"][model_name].predict(X_trans)
        prices = np.maximum(np.expm1(log_preds), 1.0)
    else:
        # Default: Ensemble Blend
        weights = bundle["weights"]
        available_models = [m for m in bundle["models"] if m in weights and weights[m] > 0]
        total_w = sum(weights[m] for m in available_models)

        if total_w > 0:
            norm_weights = {m: weights[m] / total_w for m in available_models}
            log_matrix = np.column_stack([bundle["models"][m].predict(X_trans) for m in available_models])
            w_vector = np.array([norm_weights[m] for m in available_models])
            blend_log = log_matrix @ w_vector
            prices = np.maximum(np.expm1(blend_log), 1.0)
        else:
            all_preds = np.column_stack([m.predict(X_trans) for m in bundle["models"].values()])
            prices = np.maximum(np.expm1(np.mean(all_preds, axis=1)), 1.0)

    result = pd.DataFrame({"Id": test_ids, "SalePrice": np.round(prices, 2)})
    return result
