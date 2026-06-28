from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd

from config import SETTINGS


@dataclass
class GapTeam:
    attack_home: float = SETTINGS["gap_init_attack"]
    defense_home: float = SETTINGS["gap_init_defense"]
    attack_away: float = SETTINGS["gap_init_attack"]
    defense_away: float = SETTINGS["gap_init_defense"]
    matches: int = 0
    real_stats_matches: int = 0
    fallback_matches: int = 0


def _clip(x: float) -> float:
    return float(np.clip(x, SETTINGS["gap_min"], SETTINGS["gap_max"]))


def _get(ratings: Dict[str, GapTeam], team: str) -> GapTeam:
    if team not in ratings:
        ratings[team] = GapTeam()
    return ratings[team]


def _process(row: pd.Series, side: str) -> Tuple[float, int, float]:
    # side: local/visitante
    shots = row.get(f"shots_{side}", np.nan)
    corners = row.get(f"corners_proxy_{side}", np.nan)
    goals = row.get(f"goles_{side}", 0.0)
    try:
        shots_f = float(shots)
    except Exception:
        shots_f = np.nan
    try:
        corners_f = float(corners)
    except Exception:
        corners_f = 0.0
    if not np.isnan(shots_f) and shots_f > 0:
        corners_f = 0.0 if np.isnan(corners_f) else max(0.0, corners_f)
        return shots_f + SETTINGS["gap_corner_weight"] * corners_f, 0, SETTINGS["gap_real_stats_weight"]
    return max(1.0, float(goals) * SETTINGS["gap_goal_to_process"]), 1, SETTINGS["gap_fallback_weight"]


def _snapshot(home: GapTeam, away: GapTeam, neutral: int, modo: int) -> dict:
    if int(neutral) == 1:
        ah = (home.attack_home + home.attack_away) / 2.0
        dh = (home.defense_home + home.defense_away) / 2.0
        aa = (away.attack_home + away.attack_away) / 2.0
        da = (away.defense_home + away.defense_away) / 2.0
    else:
        ah, dh = home.attack_home, home.defense_home
        aa, da = away.attack_away, away.defense_away
    return {
        "gap_ataque_neutro_local": ah,
        "gap_defensa_neutro_local": dh,
        "gap_ataque_neutro_visitante": aa,
        "gap_defensa_neutro_visitante": da,
        "gap_ratio_ataque": ah / max(aa, 1e-6),
        "gap_ratio_defensa": dh / max(da, 1e-6),
        "gap_diferencial_neto": (ah - aa) - (dh - da),
        "gap_modo": modo,
    }


GAP_FEATURE_COLS = [
    "gap_ataque_neutro_local", "gap_defensa_neutro_local",
    "gap_ataque_neutro_visitante", "gap_defensa_neutro_visitante",
    "gap_ratio_ataque", "gap_ratio_defensa", "gap_diferencial_neto", "gap_modo",
]


def calculate_gap_ratings(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    # Entrenar puede ejecutarse varias veces. Si el CSV ya tenía columnas GAP,
    # las quitamos antes de recalcular para no crear columnas duplicadas.
    work = df.copy()
    work = work.loc[:, ~work.columns.duplicated()].copy()
    work = work.drop(columns=[c for c in GAP_FEATURE_COLS if c in work.columns], errors="ignore")
    work = work.sort_values("fecha").reset_index(drop=True)
    ratings: Dict[str, GapTeam] = {}
    rows = []
    for _, r in work.iterrows():
        local = str(r["equipo_local"])
        away = str(r["equipo_visitante"])
        h = _get(ratings, local)
        a = _get(ratings, away)
        val_h, modo_h, w_h = _process(r, "local")
        val_a, modo_a, w_a = _process(r, "visitante")
        modo = 0 if (modo_h == 0 and modo_a == 0) else 1
        rows.append(_snapshot(h, a, int(r.get("neutral", 1)), modo))

        exp_h = (h.attack_home + a.defense_away) / 2.0
        diff_h = val_h - exp_h
        h.attack_home = _clip(h.attack_home + SETTINGS["gap_phi_attack"] * w_h * diff_h)
        a.defense_away = _clip(a.defense_away - SETTINGS["gap_phi_defense"] * w_h * diff_h)

        exp_a = (a.attack_away + h.defense_home) / 2.0
        diff_a = val_a - exp_a
        a.attack_away = _clip(a.attack_away + SETTINGS["gap_phi_attack"] * w_a * diff_a)
        h.defense_home = _clip(h.defense_home - SETTINGS["gap_phi_defense"] * w_a * diff_a)

        h.matches += 1; a.matches += 1
        if modo_h == 0: h.real_stats_matches += 1
        else: h.fallback_matches += 1
        if modo_a == 0: a.real_stats_matches += 1
        else: a.fallback_matches += 1
    out = pd.concat([work.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out = out.loc[:, ~out.columns.duplicated()].copy()
    model = {"ratings": {k: asdict(v) for k, v in ratings.items()}, "settings": SETTINGS}
    return out, model


def features_for_match(local: str, visitante: str, gap_model: dict, neutral: int = 1) -> dict:
    ratings = gap_model.get("ratings", {}) if isinstance(gap_model, dict) else {}
    h = GapTeam(**ratings.get(local, {})) if local in ratings else GapTeam()
    a = GapTeam(**ratings.get(visitante, {})) if visitante in ratings else GapTeam()
    modo = 0 if (h.real_stats_matches + a.real_stats_matches) >= 4 else 1
    return _snapshot(h, a, neutral, modo)
