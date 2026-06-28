from __future__ import annotations

import argparse
import json
from typing import Dict, Any, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score

from config import PATHS, SETTINGS
from consensus import combine_probabilities
from filter_certainty import apply_filter
from ml_model import prepare_matrix, _new_base_model, FEATURE_COLS


def _profit(hit: int, odds: float) -> float:
    return (odds - 1.0) if int(hit) == 1 else -1.0


def _max_losing_streak(hits: Iterable[int]) -> int:
    cur = 0
    best = 0
    for h in hits:
        if int(h) == 1:
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    return int(best)


def _safe_float(value, default=np.nan):
    try:
        if value is None:
            return default
        v = float(value)
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default


def _calibrated_fold_predict(train: pd.DataFrame, val: pd.DataFrame, feature_cols=None) -> tuple[np.ndarray, str]:
    """
    Predicción temporal real: entrena solo con partidos anteriores al año validado.
    Calibra con el último tramo del entrenamiento, nunca con el año validado.
    """
    feature_cols = feature_cols or FEATURE_COLS
    Xtr, ytr, cols = prepare_matrix(train, feature_cols)
    Xva, _, _ = prepare_matrix(val, cols)

    n = len(Xtr)
    if n < SETTINGS["min_rows_train_ml"]:
        return np.full(len(val), np.nan), "train_insuficiente"

    n_cal = max(80, int(n * SETTINGS["calibration_ratio"]))
    n_cal = min(n_cal, max(80, n // 3))
    fit_end = max(1, n - n_cal)

    base, backend = _new_base_model()
    base.fit(Xtr.iloc[:fit_end], ytr.iloc[:fit_end])

    calibrator = None
    if fit_end < n and len(set(ytr.iloc[fit_end:])) >= 2:
        raw_cal = base.predict_proba(Xtr.iloc[fit_end:])[:, 1]
        calibrator = LogisticRegression(solver="lbfgs")
        calibrator.fit(raw_cal.reshape(-1, 1), ytr.iloc[fit_end:])

    raw_val = base.predict_proba(Xva)[:, 1]
    if calibrator is not None:
        prob = calibrator.predict_proba(raw_val.reshape(-1, 1))[:, 1]
        status = f"{backend}_sigmoid_oos"
    else:
        prob = raw_val
        status = f"{backend}_sin_calibrar_oos"

    return np.clip(prob, 0.001, 0.999), status


def build_oos_predictions(start_year: int | None = None, force: bool = False) -> pd.DataFrame:
    if not PATHS["dataset"].exists():
        raise FileNotFoundError("Falta dataset_ultra.csv. Ejecuta primero: python train.py --no-statsbomb")

    out_path = PATHS["backtest_oos_predictions"]
    if out_path.exists() and not force:
        return pd.read_csv(out_path)

    df = pd.read_csv(PATHS["dataset"])
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)
    df = df[pd.to_datetime(df["fecha"]).dt.year >= SETTINGS["train_min_year"]].copy().reset_index(drop=True)

    years = sorted(df["fecha"].dt.year.unique())
    start_year = int(start_year or max(2018, SETTINGS["train_min_year"] + 3))

    rows = []
    for year in years:
        if year < start_year:
            continue
        train = df[df["fecha"].dt.year < year].copy()
        val = df[df["fecha"].dt.year == year].copy()
        if len(val) < 30 or len(train) < SETTINGS["min_rows_train_ml"]:
            continue

        print(f"[OOS] Año {year}: train={len(train)} val={len(val)}")
        p_ml, ml_status = _calibrated_fold_predict(train, val)

        for i, (_, r) in enumerate(val.iterrows()):
            f = r.to_dict()
            p_i = None if np.isnan(p_ml[i]) else float(p_ml[i])
            cons = combine_probabilities(f, p_i)
            rows.append({
                "fecha": r.get("fecha"),
                "year": int(year),
                "equipo_local": r.get("equipo_local"),
                "equipo_visitante": r.get("equipo_visitante"),
                "real_over25": int(r.get("over25", 0)),
                "p_over_ml_oos": p_i,
                "p_over_poisson_base": cons.get("p_over_poisson_base"),
                "p_over_poisson_gap": cons.get("p_over_poisson_gap"),
                "p_over_final": cons.get("p_over_final"),
                "confianza_modelo": cons.get("confidence_model"),
                "discrepancia_modelos": cons.get("discrepancy"),
                "votes_over": cons.get("votes_over"),
                "votes_under": cons.get("votes_under"),
                "modelos_usados": cons.get("models_used"),
                "ml_status": ml_status,
                "local_partidos_previos": r.get("local_partidos_previos", 0),
                "visitante_partidos_previos": r.get("visitante_partidos_previos", 0),
                "varianza_goles_local_8": r.get("varianza_goles_local_8", 0),
                "varianza_goles_visit_8": r.get("varianza_goles_visit_8", 0),
            })

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False, encoding="utf-8")
    print("Guardado OOS:", out_path)
    return out


def _model_probs_from_row(row: pd.Series) -> List[Tuple[str, float]]:
    vals = []
    for label, col in [
        ("poisson_base", "p_over_poisson_base"),
        ("poisson_gap", "p_over_poisson_gap"),
        ("ml", "p_over_ml_oos"),
    ]:
        v = _safe_float(row.get(col), np.nan)
        if not np.isnan(v):
            vals.append((label, float(np.clip(v, 0.001, 0.999))))
    return vals


def _directional_ok(row: pd.Series, margin: float) -> bool:
    """
    Regla FIX 3:
    - Si la apuesta final es UNDER, todos los modelos deben estar por debajo de 0.50-margin.
    - Si la apuesta final es OVER, todos los modelos deben estar por encima de 0.50+margin.
    """
    p_final = _safe_float(row.get("p_over_final"), 0.5)
    probs = _model_probs_from_row(row)
    if not probs:
        return False
    under_ceiling = 0.50 - float(margin)
    over_floor = 0.50 + float(margin)
    if p_final < 0.5:
        return max(v for _, v in probs) < under_ceiling
    return min(v for _, v in probs) >= over_floor


def _evaluate_threshold(
    df: pd.DataFrame,
    confidence_threshold: float,
    max_discrepancy: float | None = None,
    odds: float = 1.90,
    filter_mode: str = "directional_consensus",
    directional_margin: float = 0.05,
) -> Dict[str, Any]:
    work = df.copy()
    work["confianza_modelo"] = pd.to_numeric(work["confianza_modelo"], errors="coerce")
    work["discrepancia_modelos"] = pd.to_numeric(work["discrepancia_modelos"], errors="coerce").fillna(0.0)
    work["p_over_final"] = pd.to_numeric(work["p_over_final"], errors="coerce")

    min_matches = SETTINGS["min_team_matches"]
    base_ok = (
        (work["local_partidos_previos"].astype(float) >= min_matches) &
        (work["visitante_partidos_previos"].astype(float) >= min_matches) &
        (work["confianza_modelo"] >= confidence_threshold) &
        (work[["varianza_goles_local_8", "varianza_goles_visit_8"]].astype(float).max(axis=1) <= 6.0)
    )

    mode = str(filter_mode)
    if mode == "legacy_discrepancy":
        model_ok = work["discrepancia_modelos"] <= float(max_discrepancy)
    elif mode == "hybrid_both":
        directional = work.apply(lambda r: _directional_ok(r, directional_margin), axis=1)
        model_ok = directional & (work["discrepancia_modelos"] <= float(max_discrepancy))
    else:
        # Nuevo modo recomendado: contradicción direccional, no diferencia absoluta.
        model_ok = work.apply(lambda r: _directional_ok(r, directional_margin), axis=1)

    ok = base_ok & model_ok
    valid = work[ok].copy()
    n_all = len(work)
    n_bets = len(valid)

    out_base = {
        "filter_mode": mode,
        "confidence_threshold": round(float(confidence_threshold), 3),
        "directional_margin": round(float(directional_margin), 3),
        "under_confirm_ceiling": round(0.50 - float(directional_margin), 3),
        "over_confirm_floor": round(0.50 + float(directional_margin), 3),
        "max_discrepancy": np.nan if max_discrepancy is None else round(float(max_discrepancy), 3),
    }

    if n_bets == 0:
        return {
            **out_base,
            "bets": 0,
            "coverage": 0.0,
            "accuracy": np.nan,
            "roi_185": np.nan,
            "roi_190": np.nan,
            "roi_200": np.nan,
            "profit_190": 0.0,
            "max_losing_streak": np.nan,
            "over_bets": 0,
            "under_bets": 0,
            "over_accuracy": np.nan,
            "under_accuracy": np.nan,
            "avg_confidence": np.nan,
        }

    valid["pred_over"] = (valid["p_over_final"] >= 0.5).astype(int)
    valid["hit"] = (valid["pred_over"] == valid["real_over25"].astype(int)).astype(int)
    hits = valid["hit"].astype(int).tolist()
    over = valid[valid["pred_over"] == 1]
    under = valid[valid["pred_over"] == 0]

    profits = {od: valid["hit"].apply(lambda h: _profit(int(h), od)).sum() for od in [1.85, 1.90, 2.00]}
    return {
        **out_base,
        "bets": int(n_bets),
        "coverage": float(n_bets / max(n_all, 1)),
        "accuracy": float(valid["hit"].mean()),
        "roi_185": float(profits[1.85] / n_bets),
        "roi_190": float(profits[1.90] / n_bets),
        "roi_200": float(profits[2.00] / n_bets),
        "profit_190": float(profits[1.90]),
        "max_losing_streak": _max_losing_streak(hits),
        "over_bets": int(len(over)),
        "under_bets": int(len(under)),
        "over_accuracy": float(over["hit"].mean()) if len(over) else np.nan,
        "under_accuracy": float(under["hit"].mean()) if len(under) else np.nan,
        "avg_confidence": float(valid["confianza_modelo"].mean()),
    }


def run_threshold_grid(oos: pd.DataFrame, apply_best: bool = False) -> pd.DataFrame:
    confidence_grid = [0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70]
    discrepancy_grid = [0.12, 0.16, 0.18, 0.20, 0.22, 0.26]
    margin_grid = [0.03, 0.04, 0.05, 0.06, 0.07]

    rows = []

    # NUEVO: filtro por consenso direccional 45/55, 46/54, etc.
    for th in confidence_grid:
        for margin in margin_grid:
            rows.append(_evaluate_threshold(
                oos,
                confidence_threshold=th,
                filter_mode="directional_consensus",
                directional_margin=margin,
            ))

    # ÚTIL CONSERVADO: filtro antiguo por discrepancia absoluta para comparar.
    for th in confidence_grid:
        for disc in discrepancy_grid:
            rows.append(_evaluate_threshold(
                oos,
                confidence_threshold=th,
                max_discrepancy=disc,
                filter_mode="legacy_discrepancy",
            ))

    # Modo híbrido: exige consenso direccional y además discrepancia absoluta.
    # Se conserva para análisis, pero no se prioriza salvo que el backtest lo demuestre claramente.
    for th in [0.58, 0.60, 0.62, 0.64]:
        for margin in [0.04, 0.05, 0.06]:
            for disc in [0.16, 0.18, 0.22]:
                rows.append(_evaluate_threshold(
                    oos,
                    confidence_threshold=th,
                    max_discrepancy=disc,
                    filter_mode="hybrid_both",
                    directional_margin=margin,
                ))

    grid = pd.DataFrame(rows).sort_values(["roi_190", "accuracy", "bets"], ascending=[False, False, False])
    grid.to_csv(PATHS["threshold_grid"], index=False, encoding="utf-8")

    min_bets = int(SETTINGS.get("min_bets_for_auto_threshold", 150))
    cov_min = float(SETTINGS.get("target_coverage_min", 0.05))
    cov_max = float(SETTINGS.get("target_coverage_max", 0.25))

    candidates = grid[(grid["bets"] >= min_bets) & (grid["coverage"] >= cov_min) & (grid["coverage"] <= cov_max)].copy()
    if SETTINGS.get("prefer_directional_consensus", True):
        directional_candidates = candidates[candidates["filter_mode"] == "directional_consensus"].copy()
        if not directional_candidates.empty:
            candidates = directional_candidates
    if candidates.empty:
        candidates = grid[grid["bets"] >= min_bets].copy()
    if candidates.empty:
        candidates = grid.copy()

    # Score conservador: ROI primero, luego accuracy, y penaliza rachas largas.
    candidates["score"] = (
        candidates["roi_190"].fillna(-9) * 1.0 +
        candidates["accuracy"].fillna(0) * 0.25 +
        np.minimum(candidates["coverage"].fillna(0), 0.25) * 0.08 -
        candidates["max_losing_streak"].fillna(99) * 0.002
    )
    best = candidates.sort_values(["score", "roi_190", "accuracy"], ascending=[False, False, False]).iloc[0].to_dict()

    rec = {
        "source": "backtest_oos_grid_fix3_directional",
        "filter_mode": str(best["filter_mode"]),
        "confidence_threshold": float(best["confidence_threshold"]),
        "confidence_threshold_strict": float(max(float(best["confidence_threshold"]), SETTINGS["confidence_threshold_strict"])),
        "directional_margin": float(best.get("directional_margin", SETTINGS.get("directional_margin", 0.05))),
        "under_confirm_ceiling": float(best.get("under_confirm_ceiling", 0.45)),
        "over_confirm_floor": float(best.get("over_confirm_floor", 0.55)),
        "max_model_discrepancy": None if pd.isna(best.get("max_discrepancy", np.nan)) else float(best["max_discrepancy"]),
        "bets": int(best["bets"]),
        "coverage": float(best["coverage"]),
        "accuracy": float(best["accuracy"]),
        "roi_190": float(best["roi_190"]),
        "profit_190": float(best["profit_190"]),
        "max_losing_streak": int(best["max_losing_streak"]),
        "note": "FIX3: usa consenso direccional 45/55 por defecto. La discrepancia absoluta queda como comparativa legacy, no bloqueo principal.",
    }

    if apply_best:
        PATHS["recommended_thresholds"].write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nBACKTEST OOS PROFESIONAL — FIX 3")
    print("Partidos OOS:", len(oos))
    try:
        print("Brier final:", f"{brier_score_loss(oos['real_over25'].astype(int), oos['p_over_final'].astype(float)):.4f}")
        print("Accuracy 0.5 sin filtro:", f"{accuracy_score(oos['real_over25'].astype(int), (oos['p_over_final'].astype(float)>=0.5).astype(int)):.1%}")
        print("ROC AUC final:", f"{roc_auc_score(oos['real_over25'].astype(int), oos['p_over_final'].astype(float)):.4f}")
    except Exception:
        pass

    cols = [
        "filter_mode", "confidence_threshold", "directional_margin",
        "under_confirm_ceiling", "over_confirm_floor", "max_discrepancy",
        "bets", "coverage", "accuracy", "roi_190", "profit_190",
        "max_losing_streak", "over_bets", "under_bets", "over_accuracy", "under_accuracy", "avg_confidence"
    ]
    print("\nTop 10 umbrales/modos:")
    print(grid.head(10)[cols].to_string(index=False))

    print("\nTop 10 SOLO directional_consensus:")
    directional = grid[grid["filter_mode"] == "directional_consensus"]
    print(directional.head(10)[cols].to_string(index=False))

    print("\nRECOMENDADO:")
    for k, v in rec.items():
        print(f"  {k}: {v}")
    if apply_best:
        print("\nAplicado en:", PATHS["recommended_thresholds"])
        print("predict.py usará esos umbrales automáticamente.")
    else:
        print("\nPara aplicar el recomendado ejecuta: python backtest.py --apply-best")
    print("\nCSV OOS:", PATHS["backtest_oos_predictions"])
    print("CSV grid:", PATHS["threshold_grid"])
    return grid


def quick_backtest() -> pd.DataFrame:
    if not PATHS["dataset"].exists():
        raise FileNotFoundError("Falta dataset. Ejecuta python train.py primero.")
    df = pd.read_csv(PATHS["dataset"])
    rows = []
    for _, r in df.iterrows():
        f = r.to_dict()
        cons = combine_probabilities(f, None)
        filt = apply_filter(f, cons, ml_status="ml_no_usado_backtest_rapido")
        if filt["decision"] == "NO APOSTAR":
            hit = np.nan
        else:
            pred_over = 1 if filt["decision"].startswith("OVER") else 0
            hit = int(pred_over == int(r["over25"]))
        rows.append({
            "fecha": r.get("fecha"),
            "equipo_local": r.get("equipo_local"),
            "equipo_visitante": r.get("equipo_visitante"),
            "real_over25": int(r.get("over25", 0)),
            "decision": filt["decision"],
            "p_over_final": filt["p_over_final"],
            "confianza_modelo": filt["confidence_model"],
            "hit": hit,
            "bloqueos": filt["bloqueos"],
        })
    out = pd.DataFrame(rows)
    out.to_csv(PATHS["backtest_results"], index=False, encoding="utf-8")
    valid = out[out["decision"] != "NO APOSTAR"]
    print("BACKTEST RÁPIDO POISSON CONSENSUS")
    print("Partidos:", len(out))
    print("Apuestas:", len(valid), f"coverage={len(valid)/max(len(out),1):.1%}")
    if len(valid):
        print("Accuracy apuestas:", f"{valid['hit'].mean():.1%}")
    print("Guardado:", PATHS["backtest_results"])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Backtest rápido antiguo sin ML OOS.")
    parser.add_argument("--force", action="store_true", help="Recalcula predicciones OOS aunque ya existan.")
    parser.add_argument("--apply-best", action="store_true", help="Guarda los mejores umbrales para que predict.py los use.")
    parser.add_argument("--start-year", type=int, default=None, help="Primer año de validación OOS.")
    args = parser.parse_args()

    if args.quick:
        quick_backtest()
        return

    oos = build_oos_predictions(start_year=args.start_year, force=args.force)
    run_threshold_grid(oos, apply_best=args.apply_best)


if __name__ == "__main__":
    main()
