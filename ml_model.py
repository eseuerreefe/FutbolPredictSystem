from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import SETTINGS, PATHS

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except Exception:
    lgb = None
    HAS_LIGHTGBM = False

TARGET = "over25"

FEATURE_COLS = [
    # Base robusta
    "elo_local_pre", "elo_visitante_pre", "elo_diff",
    "local_partidos_previos", "visitante_partidos_previos",
    "local_goles_for_forma", "visitante_goles_for_forma",
    "local_goles_contra_forma", "visitante_goles_contra_forma",
    "local_total_goles_forma", "visitante_total_goles_forma",
    "local_over25_forma", "visitante_over25_forma",
    "local_long_over25_rate", "visitante_long_over25_rate",
    "local_puntos_forma", "visitante_puntos_forma",
    "varianza_goles_local_8", "varianza_goles_visit_8",
    "dias_descanso_local", "dias_descanso_visitante", "neutral", "peso_torneo",
    "fase_codigo", "phase_penalty", "altitud_sede", "altitude_penalty",
    # Stats opcionales reales
    "statsbomb_available", "shots_local", "shots_visitante",
    "shots_on_target_local", "shots_on_target_visitante",
    "xg_local", "xg_visitante",
    "local_shots_for_forma", "visitante_shots_for_forma",
    "local_shots_contra_forma", "visitante_shots_contra_forma",
    "local_xg_for_forma", "local_xg_contra_forma",
    "visitante_xg_for_forma", "visitante_xg_contra_forma",
    "varianza_shots_local_8", "varianza_shots_visit_8",
    # GAP + Poisson
    "gap_ataque_neutro_local", "gap_defensa_neutro_local",
    "gap_ataque_neutro_visitante", "gap_defensa_neutro_visitante",
    "gap_ratio_ataque", "gap_ratio_defensa", "gap_diferencial_neto", "gap_modo",
    "lambda_total_base", "p_over_poisson_base",
    "lambda_local_gap", "lambda_visitante_gap", "lambda_total_gap", "p_over_poisson_gap",
    # H2H
    "h2h_total_goles_promedio", "h2h_over25_rate",
]


class SigmoidCalibratedModel:
    def __init__(self, base_model, calibrator=None, feature_cols: List[str] | None = None, backend: str = "unknown"):
        self.base_model = base_model
        self.calibrator = calibrator
        self.feature_cols = feature_cols or []
        self.backend = backend

    def predict_proba(self, X: pd.DataFrame):
        X = X[self.feature_cols].copy()
        p = self.base_model.predict_proba(X)[:, 1]
        if self.calibrator is not None:
            p = self.calibrator.predict_proba(p.reshape(-1, 1))[:, 1]
        p = np.clip(p, 0.001, 0.999)
        return np.column_stack([1.0 - p, p])


def _new_base_model():
    if HAS_LIGHTGBM:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", lgb.LGBMClassifier(
                n_estimators=280,
                learning_rate=0.025,
                max_depth=3,
                num_leaves=9,
                min_child_samples=35,
                subsample=0.85,
                colsample_bytree=0.75,
                reg_alpha=0.2,
                reg_lambda=1.5,
                random_state=SETTINGS["random_state"],
                verbose=-1,
            )),
        ]), "lightgbm"
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.035,
            max_depth=2,
            random_state=SETTINGS["random_state"],
        )),
    ]), "gradient_boosting"


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def prepare_matrix(df: pd.DataFrame, feature_cols: List[str] | None = None) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    work = df.copy()
    # Defensa anti-fallos: si por reentrenar varias veces hay columnas duplicadas,
    # pandas devuelve un DataFrame en work[c] y pd.to_numeric rompe.
    work = work.loc[:, ~work.columns.duplicated()].copy()
    feature_cols = _dedupe_keep_order(list(feature_cols or FEATURE_COLS))

    for c in feature_cols:
        if c not in work.columns:
            work[c] = np.nan
        col = work[c]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        col = pd.to_numeric(col, errors="coerce")
        if col.isna().all():
            col = pd.Series(0.0, index=work.index)
        work[c] = col

    X = work.loc[:, feature_cols].astype(float)
    if TARGET in work.columns:
        y_col = work[TARGET]
        if isinstance(y_col, pd.DataFrame):
            y_col = y_col.iloc[:, 0]
        y = pd.to_numeric(y_col, errors="coerce").fillna(0).astype(int)
    else:
        y = pd.Series(dtype=int)
    return X, y, feature_cols


def _safe_metrics(y_true, prob) -> Dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    out = {
        "brier": float(brier_score_loss(y_true, prob)),
        "accuracy": float(accuracy_score(y_true, pred)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
    except Exception:
        out["roc_auc"] = np.nan
    return out


def temporal_cv(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    work = df.copy().sort_values("fecha").reset_index(drop=True)
    years = sorted(pd.to_datetime(work["fecha"]).dt.year.dropna().unique())
    rows = []
    for val_year in years:
        if val_year < max(2018, SETTINGS["train_min_year"] + 3):
            continue
        train = work[pd.to_datetime(work["fecha"]).dt.year < val_year]
        val = work[pd.to_datetime(work["fecha"]).dt.year == val_year]
        if len(train) < SETTINGS["min_rows_train_ml"] or len(val) < 50:
            continue
        Xtr, ytr, cols = prepare_matrix(train, feature_cols)
        Xva, yva, _ = prepare_matrix(val, cols)
        model, backend = _new_base_model()
        model.fit(Xtr, ytr)
        prob = model.predict_proba(Xva)[:, 1]
        m = _safe_metrics(yva, prob)
        conf = np.maximum(prob, 1.0 - prob)
        rows.append({
            "val_year": int(val_year),
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "backend": backend,
            "coverage_conf_58": float(np.mean(conf >= 0.58)),
            "coverage_conf_64": float(np.mean(conf >= 0.64)),
            **m,
        })
    return pd.DataFrame(rows)


def train_ml(df: pd.DataFrame) -> Dict[str, Any]:
    train_df = df[
        (df["local_partidos_previos"] >= 3) &
        (df["visitante_partidos_previos"] >= 3) &
        (pd.to_datetime(df["fecha"]).dt.year >= SETTINGS["train_min_year"])
    ].copy().sort_values("fecha").reset_index(drop=True)

    if len(train_df) < SETTINGS["min_rows_train_ml"]:
        return {"trained": False, "reason": f"Filas insuficientes: {len(train_df)}"}

    X, y, cols = prepare_matrix(train_df)
    cv = temporal_cv(train_df, cols)
    cv.to_csv(PATHS["cv_results"], index=False)

    n = len(train_df)
    n_cal = max(80, int(n * SETTINGS["calibration_ratio"]))
    n_cal = min(n_cal, max(80, n // 3))
    fit_idx = list(range(0, n - n_cal))
    cal_idx = list(range(n - n_cal, n))
    if len(fit_idx) < 300:
        fit_idx = list(range(n))
        cal_idx = []

    base, backend = _new_base_model()
    base.fit(X.iloc[fit_idx], y.iloc[fit_idx])

    calibrator = None
    cal_metrics = {}
    if cal_idx:
        raw_cal = base.predict_proba(X.iloc[cal_idx])[:, 1]
        calibrator = LogisticRegression(solver="lbfgs")
        calibrator.fit(raw_cal.reshape(-1, 1), y.iloc[cal_idx])
        cal_prob = calibrator.predict_proba(raw_cal.reshape(-1, 1))[:, 1]
        cal_metrics = _safe_metrics(y.iloc[cal_idx], cal_prob)
        cal_metrics["calibration_rows"] = len(cal_idx)
        cal_metrics["calibration_method"] = "sigmoid_holdout_temporal"
    else:
        cal_metrics["calibration_method"] = "none"

    # Entrenamos base final con todo y mantenemos calibrador aprendido en el holdout temporal.
    final_base, backend = _new_base_model()
    final_base.fit(X, y)
    model = SigmoidCalibratedModel(final_base, calibrator, cols, backend=backend)

    # Importancias si existen
    try:
        clf = final_base.named_steps.get("clf")
        imp = getattr(clf, "feature_importances_", None)
        if imp is not None:
            pd.DataFrame({"feature": cols, "importance": imp}).sort_values("importance", ascending=False).to_csv(PATHS["feature_importance"], index=False)
    except Exception:
        pass

    report = {
        "trained": True,
        "rows": int(len(train_df)),
        "feature_cols": cols,
        "backend": backend,
        "cv_rows": cv.to_dict(orient="records"),
        "calibration": cal_metrics,
    }
    return {"trained": True, "model": model, "report": report, "feature_cols": cols}


def predict_ml_proba(artifact: Dict[str, Any], features: Dict[str, Any]) -> Tuple[float | None, str]:
    ml = artifact.get("ml", {}) if isinstance(artifact, dict) else {}
    if not ml or not ml.get("trained"):
        return None, "ml_no_entrenado"
    try:
        model = ml["model"]
        cols = ml.get("feature_cols", getattr(model, "feature_cols", FEATURE_COLS))
        X = pd.DataFrame([{c: features.get(c, np.nan) for c in cols}])
        return float(model.predict_proba(X)[0, 1]), "ok"
    except Exception as exc:
        return None, repr(exc)
