from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from config import URLS, PATHS, RAW_DIR, SETTINGS
from utils import normalize_team


def fetch_url(url: str, cache_path: Path, force: bool = False, timeout: int = 35) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not force:
        return cache_path
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    cache_path.write_bytes(r.content)
    return cache_path


def load_international_results(force: bool = False) -> pd.DataFrame:
    path = fetch_url(URLS["international_results"], PATHS["raw_results"], force=force)
    df = pd.read_csv(path)
    required = ["date", "home_team", "away_team", "home_score", "away_score", "tournament"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"international_results no tiene columnas: {missing}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"]).copy()
    df["home_team"] = df["home_team"].map(normalize_team)
    df["away_team"] = df["away_team"].map(normalize_team)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df[(df["home_team"] != df["away_team"]) & (df["date"].dt.year >= SETTINGS["min_year"])]
    df = df[df["date"].dt.year <= SETTINGS["max_year_train"]]
    return df.sort_values("date").reset_index(drop=True)


def try_build_statsbomb(force: bool = False, max_matches: Optional[int] = None, verbose: bool = True) -> pd.DataFrame:
    if not SETTINGS.get("use_statsbomb", True):
        return pd.DataFrame()
    max_matches = int(max_matches or SETTINGS["max_statsbomb_matches"])
    try:
        comp_path = fetch_url(URLS["statsbomb_competitions"], PATHS["statsbomb_competitions"], force=force)
        comps = json.loads(comp_path.read_text(encoding="utf-8"))
    except Exception as exc:
        if verbose:
            print(f"[StatsBomb] No disponible: {exc}")
        return pd.DataFrame()

    wanted = []
    for c in comps:
        name = str(c.get("competition_name", ""))
        if any(x in name for x in ["FIFA World Cup", "UEFA Euro", "Copa America", "Copa América"]):
            wanted.append(c)

    rows = []
    count = 0
    base_url = URLS["statsbomb_base"]
    for c in wanted:
        if count >= max_matches:
            break
        comp_id = c.get("competition_id")
        season_id = c.get("season_id")
        matches_url = f"{base_url}/matches/{comp_id}/{season_id}.json"
        matches_cache = RAW_DIR / "statsbomb_matches" / f"{comp_id}_{season_id}.json"
        try:
            mp = fetch_url(matches_url, matches_cache, force=force)
            matches = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            continue

        for m in matches:
            if count >= max_matches:
                break
            try:
                match_id = m["match_id"]
                home = normalize_team(m["home_team"]["home_team_name"])
                away = normalize_team(m["away_team"]["away_team_name"])
                match_date = pd.to_datetime(m.get("match_date"), errors="coerce")
                if pd.isna(match_date):
                    continue
                events_url = f"{base_url}/events/{match_id}.json"
                events_cache = RAW_DIR / "statsbomb_events" / f"{match_id}.json"
                ep = fetch_url(events_url, events_cache, force=force, timeout=50)
                events = json.loads(ep.read_text(encoding="utf-8"))
            except Exception:
                continue

            shots = {home: 0, away: 0}
            sot = {home: 0, away: 0}
            xg = {home: 0.0, away: 0.0}
            corners = {home: 0, away: 0}
            for e in events:
                team = normalize_team(e.get("team", {}).get("name", ""))
                if team not in shots:
                    continue
                typ = e.get("type", {}).get("name", "")
                if typ == "Shot" and e.get("period", 0) != 5:
                    shots[team] += 1
                    sh = e.get("shot", {})
                    xg[team] += float(sh.get("statsbomb_xg", 0.0) or 0.0)
                    outcome = sh.get("outcome", {}).get("name", "")
                    if outcome in {"Goal", "Saved", "Saved to Post"}:
                        sot[team] += 1
                # StatsBomb corners pueden venir como Pass desde corner o Starting XI no. Marcamos aproximación conservadora.
                play = e.get("play_pattern", {}).get("name", "")
                if play == "From Corner" and typ in {"Pass", "Shot"}:
                    # Cuenta como presencia de ataque desde córner, no córners exactos.
                    corners[team] += 1
            rows.append({
                "fecha": match_date.normalize(),
                "equipo_local": home,
                "equipo_visitante": away,
                "shots_local": shots[home],
                "shots_visitante": shots[away],
                "shots_on_target_local": sot[home],
                "shots_on_target_visitante": sot[away],
                "corners_proxy_local": corners[home],
                "corners_proxy_visitante": corners[away],
                "xg_local": xg[home],
                "xg_visitante": xg[away],
                "statsbomb_match_id": match_id,
            })
            count += 1
            if verbose and count % 50 == 0:
                print(f"[StatsBomb] {count} partidos procesados")
            time.sleep(0.03)
    sb = pd.DataFrame(rows)
    if not sb.empty:
        sb.to_csv(PATHS["statsbomb_match_stats"], index=False, encoding="utf-8")
    return sb
