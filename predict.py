from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from config import PATHS, SETTINGS, VENUES
from utils import normalize_team, phase_key, phase_code, phase_penalty
from gap_ratings import features_for_match
from poisson_models import base_poisson_from_features, gap_poisson_from_features
from ml_model import predict_ml_proba
from consensus import combine_probabilities
from filter_certainty import apply_filter


def load_artifact() -> dict:
    if not PATHS["artifact"].exists():
        raise FileNotFoundError("No existe el modelo. Ejecuta primero: python train.py")
    with open(PATHS["artifact"], "rb") as f:
        return pickle.load(f)


def load_state() -> pd.DataFrame:
    if not PATHS["team_state"].exists():
        raise FileNotFoundError("No existe team_state. Ejecuta primero: python build_dataset.py")
    return pd.read_csv(PATHS["team_state"])


def load_dataset_optional() -> pd.DataFrame:
    if PATHS["dataset"].exists():
        df = pd.read_csv(PATHS["dataset"])
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        return df
    return pd.DataFrame()


def _state_row(state: pd.DataFrame, team: str) -> Dict[str, Any]:
    if state.empty:
        return {}
    row = state[state["equipo"].astype(str).str.lower() == team.lower()]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def _g(d: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        v = float(d.get(key, default))
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default


def _h2h_features(df: pd.DataFrame, home: str, away: str) -> Dict[str, float]:
    if df.empty:
        return {"h2h_total_goles_promedio": np.nan, "h2h_over25_rate": np.nan}
    mask = ((df["equipo_local"] == home) & (df["equipo_visitante"] == away)) | ((df["equipo_local"] == away) & (df["equipo_visitante"] == home))
    sub = df[mask].sort_values("fecha").tail(8)
    if len(sub) < 3:
        return {"h2h_total_goles_promedio": np.nan, "h2h_over25_rate": np.nan}
    return {
        "h2h_total_goles_promedio": float(sub["total_goles"].mean()),
        "h2h_over25_rate": float(sub["over25"].mean()),
    }


def make_features(match: Dict[str, Any], artifact: dict, state: pd.DataFrame, history: pd.DataFrame) -> Dict[str, Any]:
    home = normalize_team(match.get("equipo_local") or match.get("local") or match.get("home"))
    away = normalize_team(match.get("equipo_visitante") or match.get("visitante") or match.get("away"))
    sede = str(match.get("sede", "") or "")
    fase = phase_key(match.get("fase", "Grupo"))
    neutral = int(float(match.get("neutral", 1) or 1))
    venue = VENUES.get(sede, {"altitude": float(match.get("altitud_sede", 0) or 0)})
    altitude = float(venue.get("altitude", 0.0))

    hs = _state_row(state, home)
    aw = _state_row(state, away)
    elo_home = _g(hs, "elo_actual", SETTINGS["elo_default"])
    elo_away = _g(aw, "elo_actual", SETTINGS["elo_default"])
    elo_diff = elo_home - elo_away  # neutral por defecto

    f = {
        "equipo_local": home,
        "equipo_visitante": away,
        "sede": sede,
        "fase": fase,
        "fecha": match.get("fecha", ""),
        "elo_local_pre": elo_home,
        "elo_visitante_pre": elo_away,
        "elo_diff": elo_diff,
        "local_partidos_previos": int(_g(hs, "partidos_totales", 0)),
        "visitante_partidos_previos": int(_g(aw, "partidos_totales", 0)),
        "local_goles_for_forma": _g(hs, "goles_for_forma", 0.0),
        "visitante_goles_for_forma": _g(aw, "goles_for_forma", 0.0),
        "local_goles_contra_forma": _g(hs, "goles_contra_forma", 0.0),
        "visitante_goles_contra_forma": _g(aw, "goles_contra_forma", 0.0),
        "local_total_goles_forma": _g(hs, "total_goles_forma", SETTINGS["world_avg_total_goals"]),
        "visitante_total_goles_forma": _g(aw, "total_goles_forma", SETTINGS["world_avg_total_goals"]),
        "local_over25_forma": _g(hs, "over25_forma", 0.5),
        "visitante_over25_forma": _g(aw, "over25_forma", 0.5),
        "local_long_over25_rate": _g(hs, "long_over25_rate", 0.5),
        "visitante_long_over25_rate": _g(aw, "long_over25_rate", 0.5),
        "local_puntos_forma": _g(hs, "puntos_forma", 0.0),
        "visitante_puntos_forma": _g(aw, "puntos_forma", 0.0),
        "varianza_goles_local_8": _g(hs, "varianza_goles_8", 0.0),
        "varianza_goles_visit_8": _g(aw, "varianza_goles_8", 0.0),
        "local_shots_for_forma": _g(hs, "shots_for_forma", np.nan),
        "visitante_shots_for_forma": _g(aw, "shots_for_forma", np.nan),
        "local_shots_contra_forma": _g(hs, "shots_contra_forma", np.nan),
        "visitante_shots_contra_forma": _g(aw, "shots_contra_forma", np.nan),
        "local_xg_for_forma": _g(hs, "xg_for_forma", np.nan),
        "local_xg_contra_forma": _g(hs, "xg_contra_forma", np.nan),
        "visitante_xg_for_forma": _g(aw, "xg_for_forma", np.nan),
        "visitante_xg_contra_forma": _g(aw, "xg_contra_forma", np.nan),
        "varianza_shots_local_8": _g(hs, "varianza_shots_8", 0.0),
        "varianza_shots_visit_8": _g(aw, "varianza_shots_8", 0.0),
        "dias_descanso_local": float(match.get("dias_descanso_local", 7) or 7),
        "dias_descanso_visitante": float(match.get("dias_descanso_visitante", 7) or 7),
        "neutral": neutral,
        "peso_torneo": float(match.get("peso_torneo", 1.0) or 1.0),
        "fase_codigo": phase_code(fase),
        "phase_penalty": phase_penalty(fase),
        "altitud_sede": altitude,
        "altitude_penalty": SETTINGS["altitude_lambda_penalty"] if altitude > SETTINGS["altitude_threshold"] else 0.0,
        "statsbomb_available": 0,
        "shots_local": np.nan,
        "shots_visitante": np.nan,
        "shots_on_target_local": np.nan,
        "shots_on_target_visitante": np.nan,
        "xg_local": np.nan,
        "xg_visitante": np.nan,
    }
    f.update(features_for_match(home, away, artifact.get("gap_model", {}), neutral=neutral))
    f.update(_h2h_features(history, home, away))
    f.update(base_poisson_from_features(f))
    f.update(gap_poisson_from_features(f, artifact.get("poisson_config", {})))
    return f


def predict_one(match: Dict[str, Any], artifact: dict | None = None, state: pd.DataFrame | None = None, history: pd.DataFrame | None = None) -> Dict[str, Any]:
    artifact = artifact or load_artifact()
    state = state if state is not None else load_state()
    history = history if history is not None else load_dataset_optional()
    f = make_features(match, artifact, state, history)
    p_ml, ml_status = predict_ml_proba(artifact, f)
    cons = combine_probabilities(f, p_ml)
    filt = apply_filter(f, cons, ml_status=ml_status)
    return {
        "equipo_local": f["equipo_local"],
        "equipo_visitante": f["equipo_visitante"],
        "fecha": f.get("fecha", ""),
        "sede": f.get("sede", ""),
        "fase": f.get("fase", ""),
        "decision": filt["decision"],
        "observacion_no_apuesta": filt["observacion_no_apuesta"],
        "p_over_final": round(filt["p_over_final"], 6),
        "p_under_final": round(filt["p_under_final"], 6),
        "confianza_modelo": round(filt["confidence_model"], 6),
        "confianza_apuesta": round(filt["confidence_bet"], 6),
        "p_over_ml": None if p_ml is None else round(float(p_ml), 6),
        "p_over_poisson_base": round(float(cons.get("p_over_poisson_base") or 0.5), 6),
        "p_over_poisson_gap": round(float(cons.get("p_over_poisson_gap") or 0.5), 6),
        "lambda_total_base": round(float(f.get("lambda_total_base", 0.0)), 4),
        "lambda_total_gap": round(float(f.get("lambda_total_gap", 0.0)), 4),
        "discrepancia_modelos": round(float(cons.get("discrepancy", 0.0)), 6),
        "modelos_usados": cons.get("models_used", ""),
        "elo_local": round(float(f.get("elo_local_pre", 0.0)), 1),
        "elo_visitante": round(float(f.get("elo_visitante_pre", 0.0)), 1),
        "partidos_previos_local": int(f.get("local_partidos_previos", 0)),
        "partidos_previos_visitante": int(f.get("visitante_partidos_previos", 0)),
        "checks_ok": filt["checks_ok"],
        "bloqueos": filt["bloqueos"],
        "filter_mode": filt.get("filter_mode", ""),
        "directional_margin_used": round(float(filt.get("directional_margin_used", 0.0)), 4),
        "under_confirm_ceiling_used": round(float(filt.get("under_confirm_ceiling_used", 0.0)), 4),
        "over_confirm_floor_used": round(float(filt.get("over_confirm_floor_used", 0.0)), 4),
        "threshold_source": filt.get("threshold_source", ""),
        "ml_status": ml_status,
    }


def print_report(r: Dict[str, Any]) -> None:
    print("═" * 54)
    print("PREDICCIÓN OVER25 ULTRA FREE")
    print("═" * 54)
    print(f"PARTIDO: {r['equipo_local']} vs {r['equipo_visitante']}")
    print(f"Fecha/Sede/Fase: {r.get('fecha','')} | {r.get('sede','')} | {r.get('fase','')}")
    print()
    print("[MODELOS]")
    print(f"Poisson base Over: {r['p_over_poisson_base']*100:.1f}% | λ total {r['lambda_total_base']:.2f}")
    print(f"Poisson GAP  Over: {r['p_over_poisson_gap']*100:.1f}% | λ total {r['lambda_total_gap']:.2f}")
    ml_txt = "N/A" if r["p_over_ml"] is None else f"{r['p_over_ml']*100:.1f}%"
    print(f"ML calibrado Over: {ml_txt}")
    print(f"Final Over:        {r['p_over_final']*100:.1f}%")
    print(f"Final Under:       {r['p_under_final']*100:.1f}%")
    print(f"Discrepancia:      {r['discrepancia_modelos']:.3f}")
    print()
    print("[FILTRO]")
    print(f"RESULTADO FINAL:      {r['decision']}")
    print(f"OBSERVACIÓN BRUTA:    {r['observacion_no_apuesta']}")
    print(f"CONFIANZA MODELO:     {r['confianza_modelo']*100:.1f}%")
    print(f"CONFIANZA APUESTA:    {r['confianza_apuesta']*100:.1f}%")
    print(f"Modo filtro:          {r.get('filter_mode', '')}")
    if r.get("filter_mode") == "directional_consensus":
        print(f"Regla direccional:    UNDER si modelos < {r.get('under_confirm_ceiling_used', 0)*100:.1f}% | OVER si modelos >= {r.get('over_confirm_floor_used', 0)*100:.1f}%")
    print(f"Checks OK:            {r['checks_ok']}")
    print(f"Bloqueos:             {r['bloqueos'] or 'ninguno'}")
    print("═" * 54)


def predict_csv(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    artifact = load_artifact()
    state = load_state()
    history = load_dataset_optional()
    inp = pd.read_csv(input_path)
    rows = [predict_one(row.to_dict(), artifact, state, history) for _, row in inp.iterrows()]
    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False, encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("local", nargs="?")
    parser.add_argument("visitante", nargs="?")
    parser.add_argument("--sede", default="")
    parser.add_argument("--fase", default="Grupo")
    parser.add_argument("--fecha", default="")
    parser.add_argument("--neutral", default=1, type=int)
    parser.add_argument("--dias-local", default=7, type=float)
    parser.add_argument("--dias-visitante", default=7, type=float)
    parser.add_argument("--input", help="CSV con partidos")
    parser.add_argument("--output", default="outputs/predicciones_ultra.csv")
    args = parser.parse_args()

    if args.input:
        out = predict_csv(args.input, args.output)
        print(out.to_string(index=False))
        print("\nGuardado:", args.output)
        return
    if not args.local or not args.visitante:
        raise SystemExit("Uso: python predict.py \"South Africa\" \"Canada\" --fase Dieciseisavos --sede \"Los Angeles\"")
    r = predict_one({
        "equipo_local": args.local,
        "equipo_visitante": args.visitante,
        "sede": args.sede,
        "fase": args.fase,
        "fecha": args.fecha,
        "neutral": args.neutral,
        "dias_descanso_local": args.dias_local,
        "dias_descanso_visitante": args.dias_visitante,
    })
    print_report(r)


if __name__ == "__main__":
    main()
