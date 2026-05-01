"""
model.py — ML: predice margen % y días de cierre por venta
Modelos: Random Forest + Gradient Boosting para dos targets
Autora: Micaela Feriale
"""
import pandas as pd
import numpy as np
import pickle, os, logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

CAT_COLS = ["marca","tipo","segmento","financiacion","origen_lead","zona","senioridad"]
NUM_FEATS = [
    "precio_lista","descuento_pct","num_accesorios","permuta",
    "test_drive","segmento_num","es_premium_luxury","mes","año",
]


def encode(df):
    df = df.copy()
    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        df[col+"_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def get_X(df):
    enc_cols = [c+"_enc" for c in CAT_COLS]
    all_feats = NUM_FEATS + enc_cols
    return df[[f for f in all_feats if f in df.columns]].copy()


def eval_model(y_true, y_pred, name):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    log.info(f"  {name:<28} MAE={mae:>8.3f}  RMSE={rmse:>8.3f}  R²={r2:.4f}  MAPE={mape:.1f}%")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def train(df):
    log.info("Preparando features...")
    df_enc, encoders = encode(df)
    df_enc["permuta"] = df_enc["permuta"].astype(int)
    df_enc["test_drive"] = df_enc["test_drive"].astype(int)

    X = get_X(df_enc)
    y_margin = df_enc["margen_pct"]
    y_days   = df_enc["dias_cierre"]

    log.info(f"Features: {X.shape[1]} | Registros: {len(X):,}")

    results = {"margen": [], "dias": []}
    models  = {}
    fi_dict = {}

    for target_name, y in [("margen_pct", y_margin), ("dias_cierre", y_days)]:
        log.info(f"\n── TARGET: {target_name} ──")
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

        # Ridge baseline
        ridge = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=5.0))])
        ridge.fit(Xtr, ytr)
        r = eval_model(yte, ridge.predict(Xte), "Ridge")
        r["target"] = target_name
        results[target_name.split("_")[0]].append(r)

        # Random Forest
        rf = RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42)
        rf.fit(Xtr, ytr)
        r = eval_model(yte, rf.predict(Xte), "Random Forest")
        r["target"] = target_name
        results[target_name.split("_")[0]].append(r)

        # Gradient Boosting
        gb = GradientBoostingRegressor(n_estimators=250, max_depth=4,
                                       learning_rate=0.06, subsample=0.8, random_state=42)
        gb.fit(Xtr, ytr)
        r = eval_model(yte, gb.predict(Xte), "Gradient Boosting")
        r["target"] = target_name
        cv = cross_val_score(gb, X, y, cv=5, scoring="r2")
        log.info(f"    CV R²: {cv.mean():.4f} ± {cv.std():.4f}")
        results[target_name.split("_")[0]].append(r)

        key = f"gb_{target_name}"
        models[key] = {"model": gb, "Xte": Xte, "yte": yte, "ypred": gb.predict(Xte)}
        fi_dict[key] = pd.DataFrame({"feature": X.columns, "importance": gb.feature_importances_}).sort_values("importance", ascending=False)

    results_df = {k: pd.DataFrame(v).sort_values("R2", ascending=False) for k, v in results.items()}
    for k, df_r in results_df.items():
        log.info(f"\nMejor modelo para {k}: {df_r.iloc[0]['model']} — R²={df_r.iloc[0]['R2']:.4f}")

    arts = {"models": models, "encoders": encoders, "features": list(X.columns),
            "results": results_df, "feature_importance": fi_dict}
    with open(os.path.join(MODELS_DIR, "artifacts.pkl"), "wb") as f:
        pickle.dump(arts, f)

    pd.concat(results_df.values()).to_csv(os.path.join(MODELS_DIR, "model_results.csv"), index=False)
    fi_dict["gb_margen_pct"].to_csv(os.path.join(MODELS_DIR, "fi_margen.csv"), index=False)
    fi_dict["gb_dias_cierre"].to_csv(os.path.join(MODELS_DIR, "fi_dias.csv"), index=False)
    log.info("Artefactos guardados.")
    return arts
