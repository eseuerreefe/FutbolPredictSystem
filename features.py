from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from config import SETTINGS, VENUES
from utils import normalize_team, tournament_weight, valid_tournament, phase_code, phase_penalty


def expected_score(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(ra - rb) / 400.0))


def elo_update(ra: float, rb: float, score_a: float, goal_diff: int, k: float) -> Tuple[float, float]:
    exp_a = expected_score(ra, rb)
    gd = max(abs(goal_diff), 1)
    margin_mult = math.log(gd + 1.0) * 1.35
    change = k * margin_mult * (score_a - exp_a)
    return ra + change, rb - change


def _mean(vals: List[float], default: float = 0.0) -> float:
    vals = [float(v) for v in vals if v is not None and not pd.isna(v)]
    return float(np.mean(vals)) if vals else default


def _var(vals: List[float], default: float = 0.0) -> float:
    vals = [float(v) for v in vals if v is not None and not pd.isna(v)]
    return float(np.var(vals)) if len(vals) >= 2 else default


def add_elo_goal_form(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy().sort_values("date").reset_index(drop=True)
    elo: Dict[str, float] = {}
    history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    last_date: Dict[str, pd.Timestamp] = {}
    h2h: Dict[tuple, List[Dict[str, float]]] = defaultdict(list)
    rows = []

    def get_elo(team: str) -> float:
        return float(elo.get(team, SETTINGS["elo_default"]))

    def form(team: str, current_date: pd.Timestamp) -> Dict[str, float]:
        hist = history.get(team, [])
        recent = hist[-SETTINGS["form_window"]:]
        long_recent = hist[-SETTINGS["long_form_window"]:]
        rest = 7.0
        if team in last_date:
            rest = max(0.0, min(60.0, float((current_date - last_date[team]).days)))

        def _decayed_mean(key: str, data: List[Dict[str, Any]], default: float) -> float:
            if not data: return default
            vals, weights = [], []
            for item in data:
                v = item.get(key)
                if v is None or pd.isna(v): continue
                # Decay exponencial: el peso decae según los días transcurridos
                days_diff = max(0, (current_date - item["date"]).days)
                w = math.exp(-days_diff / 45.0) 
                vals.append(float(v))
                weights.append(w)
            if not weights or sum(weights) == 0: return default
            return sum(v * w for v, w in zip(vals, weights)) / sum(weights)

        return {
            "matches_total_pre": len(hist),
            "form_goals_for": _decayed_mean("gf", recent, 0.0),
            "form_goals_against": _decayed_mean("ga", recent, 0.0),
            "form_total_goals": _decayed_mean("total_goals", recent, SETTINGS["world_avg_total_goals"]),
            "form_over25_rate": _decayed_mean("over25", recent, 0.50),
            "long_over25_rate": _decayed_mean("over25", long_recent, 0.50),
            "form_points": _decayed_mean("points", recent, 0.0),
            "var_goals_for": _var([x["gf"] for x in recent]),
            "var_total_goals": _var([x["total_goals"] for x in recent]),
            "rest_days": rest,
        }

    for _, r in df.iterrows():
        home = normalize_team(r["home_team"])
        away = normalize_team(r["away_team"])
        date = pd.to_datetime(r["date"])
        hs = int(r["home_score"])
        aas = int(r["away_score"])
        neutral = bool(r.get("neutral", False))
        tournament = str(r.get("tournament", ""))
        if not valid_tournament(tournament):
            continue

        rh = get_elo(home)
        ra = get_elo(away)
        home_adv = 0.0 if neutral else SETTINGS["elo_home_advantage"]
        rh_eff = rh + home_adv
        hf = form(home, date)
        af = form(away, date)
        hkey = tuple(sorted([home, away]))
        prev_h2h = h2h[hkey]
        h2h_total = _mean([x["total_goals"] for x in prev_h2h], np.nan) if len(prev_h2h) >= 3 else np.nan
        h2h_over = _mean([x["over25"] for x in prev_h2h], np.nan) if len(prev_h2h) >= 3 else np.nan

        tw = tournament_weight(tournament)
        k = SETTINGS["elo_k_base"] * (0.65 + tw)
        if hs > aas:
            score_h, points_h, points_a = 1.0, 3.0, 0.0
        elif hs < aas:
            score_h, points_h, points_a = 0.0, 0.0, 3.0
        else:
            score_h, points_h, points_a = 0.5, 1.0, 1.0
        total_goals = hs + aas
        over25 = int(total_goals >= 3)

        rows.append({
            "fecha": date,
            "equipo_local": home,
            "equipo_visitante": away,
            "goles_local": hs,
            "goles_visitante": aas,
            "total_goles": total_goals,
            "over25": over25,
            "torneo": tournament,
            "city": r.get("city", ""),
            "country": r.get("country", ""),
            "neutral": int(neutral),
            "peso_torneo": tw,
            "fase_codigo": 0,
            "phase_penalty": 0.0,
            "altitud_sede": 0.0,
            "altitude_penalty": 0.0,
            "elo_local_pre": rh,
            "elo_visitante_pre": ra,
            "elo_diff": rh_eff - ra,
            "local_partidos_previos": hf["matches_total_pre"],
            "visitante_partidos_previos": af["matches_total_pre"],
            "local_goles_for_forma": hf["form_goals_for"],
            "visitante_goles_for_forma": af["form_goals_for"],
            "local_goles_contra_forma": hf["form_goals_against"],
            "visitante_goles_contra_forma": af["form_goals_against"],
            "local_total_goles_forma": hf["form_total_goals"],
            "visitante_total_goles_forma": af["form_total_goals"],
            "local_over25_forma": hf["form_over25_rate"],
            "visitante_over25_forma": af["form_over25_rate"],
            "local_long_over25_rate": hf["long_over25_rate"],
            "visitante_long_over25_rate": af["long_over25_rate"],
            "local_puntos_forma": hf["form_points"],
            "visitante_puntos_forma": af["form_points"],
            "varianza_goles_local_8": hf["var_total_goals"],
            "varianza_goles_visit_8": af["var_total_goals"],
            "dias_descanso_local": hf["rest_days"],
            "dias_descanso_visitante": af["rest_days"],
            "h2h_total_goles_promedio": h2h_total,
            "h2h_over25_rate": h2h_over,
        })

        new_rh, new_ra = elo_update(rh_eff, ra, score_h, hs - aas, k)
        elo[home] = new_rh - home_adv
        elo[away] = new_ra
        
        # Inserción actualizada para incluir la fecha en el historial
        history[home].append({"date": date, "gf": hs, "ga": aas, "total_goals": total_goals, "over25": over25, "points": points_h})
        history[away].append({"date": date, "gf": aas, "ga": hs, "total_goals": total_goals, "over25": over25, "points": points_a})
        
        last_date[home] = date
        last_date[away] = date
        h2h[hkey].append({"total_goals": total_goals, "over25": over25})

    return pd.DataFrame(rows).sort_values("fecha").reset_index(drop=True)


def merge_statsbomb(df: pd.DataFrame, sb: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if sb is None or sb.empty:
        for col in ["shots_local", "shots_visitante", "shots_on_target_local", "shots_on_target_visitante", "corners_proxy_local", "corners_proxy_visitante", "xg_local", "xg_visitante"]:
            out[col] = np.nan
        out["statsbomb_available"] = 0
        return out
    out["fecha_key"] = pd.to_datetime(out["fecha"]).dt.normalize()
    sb = sb.copy()
    sb["fecha_key"] = pd.to_datetime(sb["fecha"]).dt.normalize()
    merged = out.merge(sb.drop(columns=["fecha"]), on=["fecha_key", "equipo_local", "equipo_visitante"], how="left")
    merged["statsbomb_available"] = merged["xg_local"].notna().astype(int)
    return merged.drop(columns=["fecha_key"])


STATS_FORM_COLS = [
    "local_shots_for_forma", "local_shots_contra_forma",
    "visitante_shots_for_forma", "visitante_shots_contra_forma",
    "local_xg_for_forma", "local_xg_contra_forma",
    "visitante_xg_for_forma", "visitante_xg_contra_forma",
    "varianza_shots_local_8", "varianza_shots_visit_8",
]


def add_stats_form(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work = work.loc[:, ~work.columns.duplicated()].copy()
    work = work.drop(columns=[c for c in STATS_FORM_COLS if c in work.columns], errors="ignore")
    work = work.sort_values("fecha").reset_index(drop=True)
    hist: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    rows = []
    for _, r in work.iterrows():
        home = r["equipo_local"]
        away = r["equipo_visitante"]
        rh = hist[home][-SETTINGS["form_window"]:]
        ra = hist[away][-SETTINGS["form_window"]:]
        rows.append({
            "local_shots_for_forma": _mean([x["shots_for"] for x in rh], np.nan),
            "local_shots_contra_forma": _mean([x["shots_against"] for x in rh], np.nan),
            "visitante_shots_for_forma": _mean([x["shots_for"] for x in ra], np.nan),
            "visitante_shots_contra_forma": _mean([x["shots_against"] for x in ra], np.nan),
            "local_xg_for_forma": _mean([x["xg_for"] for x in rh], np.nan),
            "local_xg_contra_forma": _mean([x["xg_against"] for x in rh], np.nan),
            "visitante_xg_for_forma": _mean([x["xg_for"] for x in ra], np.nan),
            "visitante_xg_contra_forma": _mean([x["xg_against"] for x in ra], np.nan),
            "varianza_shots_local_8": _var([x["shots_for"] for x in rh]),
            "varianza_shots_visit_8": _var([x["shots_for"] for x in ra]),
        })
        sl = r.get("shots_local", np.nan)
        sv = r.get("shots_visitante", np.nan)
        xgl = r.get("xg_local", np.nan)
        xgv = r.get("xg_visitante", np.nan)
        hist[home].append({"shots_for": sl, "shots_against": sv, "xg_for": xgl, "xg_against": xgv})
        hist[away].append({"shots_for": sv, "shots_against": sl, "xg_for": xgv, "xg_against": xgl})
    out = pd.concat([work.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    return out.loc[:, ~out.columns.duplicated()].copy()


def build_team_state(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    teams = sorted(set(df["equipo_local"].dropna()) | set(df["equipo_visitante"].dropna()))
    for team in teams:
        mask = (df["equipo_local"] == team) | (df["equipo_visitante"] == team)
        sub = df[mask].sort_values("fecha")
        recent = sub.tail(SETTINGS["form_window"])
        gf, ga, tg, ov, pts, shots_f, shots_a, xgf, xga = [], [], [], [], [], [], [], [], []
        for _, r in recent.iterrows():
            is_home = r["equipo_local"] == team
            g_for = float(r["goles_local"] if is_home else r["goles_visitante"])
            g_against = float(r["goles_visitante"] if is_home else r["goles_local"])
            gf.append(g_for); ga.append(g_against); tg.append(float(r["total_goles"])); ov.append(float(r["over25"]))
            pts.append(3.0 if g_for > g_against else 1.0 if g_for == g_against else 0.0)
            sf = r.get("shots_local" if is_home else "shots_visitante", np.nan)
            sa = r.get("shots_visitante" if is_home else "shots_local", np.nan)
            shots_f.append(sf); shots_a.append(sa)
            xgf.append(r.get("xg_local" if is_home else "xg_visitante", np.nan))
            xga.append(r.get("xg_visitante" if is_home else "xg_local", np.nan))
        last = sub.iloc[-1] if len(sub) else {}
        # último ELO conocido: si jugó como local usamos elo_local_pre aproximado actualizado por trayectoria: mantenemos pre final como referencia.
        elo_vals = []
        for _, r in sub.tail(5).iterrows():
            elo_vals.append(float(r["elo_local_pre"] if r["equipo_local"] == team else r["elo_visitante_pre"]))
        rows.append({
            "equipo": team,
            "elo_actual": _mean(elo_vals, SETTINGS["elo_default"]),
            "partidos_totales": int(len(sub)),
            "ultimo_partido": str(pd.to_datetime(last.get("fecha", pd.NaT)).date()) if len(sub) else "",
            "goles_for_forma": _mean(gf),
            "goles_contra_forma": _mean(ga),
            "total_goles_forma": _mean(tg, SETTINGS["world_avg_total_goals"]),
            "over25_forma": _mean(ov, 0.5),
            "long_over25_rate": _mean(list(sub.tail(SETTINGS["long_form_window"])["over25"]), 0.5) if len(sub) else 0.5,
            "puntos_forma": _mean(pts),
            "varianza_goles_8": _var(tg),
            "shots_for_forma": _mean(shots_f, np.nan),
            "shots_contra_forma": _mean(shots_a, np.nan),
            "xg_for_forma": _mean(xgf, np.nan),
            "xg_contra_forma": _mean(xga, np.nan),
            "varianza_shots_8": _var(shots_f),
        })
    return pd.DataFrame(rows).sort_values("elo_actual", ascending=False)