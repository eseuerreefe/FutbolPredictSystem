from __future__ import annotations

import math
from typing import Dict, Any

import numpy as np
import pandas as pd

from config import SETTINGS


def poisson_over25(lambda_total: float) -> float:
    lambda_total = float(np.clip(lambda_total, SETTINGS["lambda_total_min"], SETTINGS["lambda_total_max"]))
    p0 = math.exp(-lambda_total)
    p1 = p0 * lambda_total
    p2 = p1 * lambda_total / 2.0
    return float(np.clip(1.0 - (p0 + p1 + p2), 0.0, 1.0))


def _as_series(df: pd.DataFrame, col: str, default=0.0) -> pd.Series:
    """Devuelve una Series incluso si el DataFrame tiene columnas duplicadas."""
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    return pd.to_numeric(obj, errors="coerce")


def fit_poisson_config(df: pd.DataFrame) -> Dict[str, float]:
    df = df.loc[:, ~df.columns.duplicated()].copy()
    tg = _as_series(df, "total_goles", 0).dropna()
    avg_total_goals = float(tg.mean()) if len(tg) else SETTINGS["world_avg_total_goals"]
    if "shots_local" in df and "shots_visitante" in df:
        st = _as_series(df, "shots_local") + _as_series(df, "shots_visitante")
        st = st.dropna()
        conversion = float(avg_total_goals / st.mean()) if len(st) and st.mean() > 0 else 0.105
    else:
        conversion = 0.105
    return {"avg_total_goals": avg_total_goals, "process_to_goal": conversion}


def base_poisson_from_features(f: Dict[str, Any]) -> Dict[str, float]:
    avg = float(SETTINGS.get("world_avg_total_goals", 2.6))
    
    # Diferencial asimétrico de forma
    local_gf = float(f.get("local_goles_for_forma", 1.0))
    visit_ga = float(f.get("visitante_goles_contra_forma", 1.0))
    visit_gf = float(f.get("visitante_goles_for_forma", 1.0))
    local_ga = float(f.get("local_goles_contra_forma", 1.0))

    # Ponderación de ataque frente a la defensa rival
    atk_local = (local_gf + visit_ga) / max(1.0, avg)
    atk_visitante = (visit_gf + local_ga) / max(1.0, avg)

    base_total = ((atk_local + atk_visitante) / 2.0) * avg
    base_total = float(np.clip(base_total, 1.60, 3.60))

    over_rate = np.nanmean([
        float(f.get("local_over25_forma", 0.50)),
        float(f.get("visitante_over25_forma", 0.50)),
        float(f.get("local_long_over25_rate", 0.50)),
        float(f.get("visitante_long_over25_rate", 0.50)),
    ])
    if np.isnan(over_rate):
        over_rate = 0.50
        
    form_adj = (over_rate - 0.50) * 0.55
    elo_diff = abs(float(f.get("elo_diff", 0.0)))
    elo_adj = float(np.clip(elo_diff / 400.0, 0.0, 0.35))
    phase_pen = float(f.get("phase_penalty", 0.0))
    alt_pen = float(f.get("altitude_penalty", 0.0))

    lambda_total = base_total * (1.0 + form_adj) * (1.0 - 0.08 * elo_adj)
    lambda_total *= (1.0 - phase_pen) * (1.0 - alt_pen)
    lambda_total = float(np.clip(lambda_total, SETTINGS["lambda_total_min"], SETTINGS["lambda_total_max"]))
    
    return {"lambda_total_base": lambda_total, "p_over_poisson_base": poisson_over25(lambda_total)}


def gap_poisson_from_features(f: Dict[str, Any], poisson_config: Dict[str, float]) -> Dict[str, float]:
    conv = float(poisson_config.get("process_to_goal", 0.105))
    ah = float(f.get("gap_ataque_neutro_local", SETTINGS["gap_init_attack"]))
    dh = float(f.get("gap_defensa_neutro_local", SETTINGS["gap_init_defense"]))
    aa = float(f.get("gap_ataque_neutro_visitante", SETTINGS["gap_init_attack"]))
    da = float(f.get("gap_defensa_neutro_visitante", SETTINGS["gap_init_defense"]))
    proc_h = (ah + da) / 2.0
    proc_a = (aa + dh) / 2.0
    lam_h = float(np.clip(proc_h * conv, SETTINGS["lambda_min"], 2.80))
    lam_a = float(np.clip(proc_a * conv, SETTINGS["lambda_min"], 2.80))
    phase_pen = float(f.get("phase_penalty", 0.0))
    alt_pen = float(f.get("altitude_penalty", 0.0))
    lam_h *= (1.0 - phase_pen) * (1.0 - alt_pen)
    lam_a *= (1.0 - phase_pen) * (1.0 - alt_pen)
    lambda_total = float(np.clip(lam_h + lam_a, SETTINGS["lambda_total_min"], SETTINGS["lambda_total_max"]))
    return {
        "lambda_local_gap": lam_h,
        "lambda_visitante_gap": lam_a,
        "lambda_total_gap": lambda_total,
        "p_over_poisson_gap": poisson_over25(lambda_total),
    }


POISSON_FEATURE_COLS = [
    "lambda_total_base", "p_over_poisson_base",
    "lambda_local_gap", "lambda_visitante_gap", "lambda_total_gap", "p_over_poisson_gap",
]


def add_poisson_features(df: pd.DataFrame, poisson_config: Dict[str, float]) -> pd.DataFrame:
    # Evita duplicados si se reentrena sobre un dataset ya enriquecido.
    base_df = df.copy()
    base_df = base_df.loc[:, ~base_df.columns.duplicated()].copy()
    base_df = base_df.drop(columns=[c for c in POISSON_FEATURE_COLS if c in base_df.columns], errors="ignore")
    rows = []
    for _, r in base_df.iterrows():
        f = r.to_dict()
        row = {}
        row.update(base_poisson_from_features(f))
        row.update(gap_poisson_from_features(f, poisson_config))
        rows.append(row)
    out = pd.concat([base_df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    return out.loc[:, ~out.columns.duplicated()].copy()