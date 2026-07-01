"""
fetch_fixtures.py — Generador automático de telegram_fixtures_worldcup.csv
==========================================================================

Descarga los partidos del Mundial 2026 y genera el CSV que lee el bot de
Telegram.

FUENTES (en orden de prioridad):
  1. API-Sports / API-Football (v3.football.api-sports.io) → la misma API
     que usa tu otro bot (autobot900...). Usa el mismo token que ya tienes.
     Plan gratuito: 100 peticiones/día.
  2. Datos hardcodeados del Mundial 2026 como fallback total.

USO:
  # Una sola vez: registra el token (el mismo que usa tu otro bot)
  set FOOTBALL_API_KEY=tu_token_aqui            # Windows
  export FOOTBALL_API_KEY="tu_token_aqui"       # Linux/Mac

  # Ejecutar manualmente:
  python fetch_fixtures.py

  # Filtrar por días:
  python fetch_fixtures.py --days 7

  # Forzar recarga aunque el CSV sea reciente:
  python fetch_fixtures.py --force

  # Ejecutar automáticamente cada día (crontab Linux):
  0 7 * * * cd /ruta/proyecto && python fetch_fixtures.py >> logs/fetch.log 2>&1

  # O con el planificador de Windows Task Scheduler apuntando a este script.

COLUMNAS QUE GENERA (las que espera el bot):
  game_id, equipo_local, equipo_visitante, sede, fase,
  fecha, hora_espana, dias_descanso_local, dias_descanso_visitante
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, date, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_DIR   = Path(__file__).resolve().parent
OUTPUT_FILE   = PROJECT_DIR / "telegram_fixtures_worldcup.csv"
CACHE_FILE    = PROJECT_DIR / "telegram_jobs" / "fixtures_cache_raw.json"

# Token de API-Sports / API-Football (el mismo que usa tu otro bot, el que SÍ funciona).
# Se lee de FOOTBALL_API_KEY, o de FOOTBALL_DATA_TOKEN como alias por compatibilidad.
APISPORTS_TOKEN = os.environ.get("FOOTBALL_API_KEY", "") or os.environ.get("FOOTBALL_DATA_TOKEN", "")

# World Cup en API-Sports: league id = 1. Temporada/año del torneo.
WORLDCUP_LEAGUE_ID = 1
WORLDCUP_SEASON = 2026

# Zona horaria de España para convertir UTC → hora local
SPAIN_UTC_OFFSET_SUMMER = 2   # CEST (verano, aplica en el Mundial 2026)
SPAIN_UTC_OFFSET_WINTER = 1   # CET  (invierno)

# Cache: no volver a descargar si el CSV tiene menos de N horas
CACHE_MAX_AGE_HOURS = 6

# ──────────────────────────────────────────────────────────────────────────────
# MAPEOS: nombres API → nombres que usa tu sistema
# ──────────────────────────────────────────────────────────────────────────────

TEAM_NAME_MAP: Dict[str, str] = {
    # football-data.org usa estos nombres; tu sistema usa los de la derecha
    "USA":                          "United States",
    "United States":                "United States",
    "Korea Republic":               "South Korea",
    "Republic of Korea":            "South Korea",
    "IR Iran":                      "Iran",
    "Saudi Arabia":                 "Saudi Arabia",
    "Côte d'Ivoire":                "Ivory Coast",
    "Cote d'Ivoire":                "Ivory Coast",
    "DR Congo":                     "DR Congo",
    "Congo DR":                     "DR Congo",
    "Bosnia and Herzegovina":       "Bosnia and Herzegovina",
    "Bosnia & Herzegovina":         "Bosnia and Herzegovina",
    "Czechia":                      "Czech Republic",
    "North Macedonia":              "North Macedonia",
    "Cape Verde":                   "Cape Verde",
    "Cabo Verde":                   "Cape Verde",
    # los demás coinciden o se normalizan igual
}

PHASE_MAP: Dict[str, str] = {
    # football-data.org stage → fase que usa tu bot
    "GROUP_STAGE":              "Grupo",
    "ROUND_OF_16":              "Dieciseisavos",
    "LAST_16":                  "Dieciseisavos",
    "ROUND_OF_32":              "Dieciseisavos",
    "LAST_32":                  "Dieciseisavos",
    "QUARTER_FINALS":           "Cuartos",
    "SEMI_FINALS":              "Semis",
    "THIRD_PLACE":              "Tercer_puesto",
    "FINAL":                    "Final",
    # api-sports.io campo "round" (texto libre, formato "Round of 32", "Group Stage - 1", etc.)
    "Round of 32":              "Dieciseisavos",
    "Round of 16":              "Octavos",
    "Quarter-finals":           "Cuartos",
    "Semi-finals":              "Semis",
    "3rd Place Final":          "Tercer_puesto",
    "Final":                    "Final",
    # por si acaso
    "Dieciseisavos de final":   "Dieciseisavos",
    "Octavos de final":         "Octavos",
    "Octavos":                  "Octavos",
    "Cuartos de final":         "Cuartos",
    "Semifinal":                "Semis",
}

# ──────────────────────────────────────────────────────────────────────────────
# DATOS HARDCODED DE FALLBACK — Mundial 2026 (Dieciseisavos completos)
# ──────────────────────────────────────────────────────────────────────────────

FALLBACK_FIXTURES: List[Dict[str, Any]] = [
    {"game_id": "53452545", "equipo_local": "South Africa",            "equipo_visitante": "Canada",                 "sede": "Los Angeles",     "fase": "Dieciseisavos", "fecha": "2026-06-28", "hora_espana": "21:00"},
    {"game_id": "53452557", "equipo_local": "Brazil",                  "equipo_visitante": "Japan",                  "sede": "Houston",         "fase": "Dieciseisavos", "fecha": "2026-06-29", "hora_espana": "19:00"},
    {"game_id": "53452541", "equipo_local": "Germany",                 "equipo_visitante": "Paraguay",               "sede": "Boston",          "fase": "Dieciseisavos", "fecha": "2026-06-29", "hora_espana": "22:30"},
    {"game_id": "53452547", "equipo_local": "Netherlands",             "equipo_visitante": "Morocco",                "sede": "Monterrey",       "fase": "Dieciseisavos", "fecha": "2026-06-30", "hora_espana": "03:00"},
    {"game_id": "53452561", "equipo_local": "Ivory Coast",             "equipo_visitante": "Norway",                 "sede": "Dallas",          "fase": "Dieciseisavos", "fecha": "2026-06-30", "hora_espana": "19:00"},
    {"game_id": "53452543", "equipo_local": "France",                  "equipo_visitante": "Sweden",                 "sede": "Nueva York",      "fase": "Dieciseisavos", "fecha": "2026-06-30", "hora_espana": "23:00"},
    {"game_id": "53452563", "equipo_local": "Mexico",                  "equipo_visitante": "Ecuador",                "sede": "Ciudad de Mexico","fase": "Dieciseisavos", "fecha": "2026-07-01", "hora_espana": "03:00"},
    {"game_id": "53452565", "equipo_local": "England",                 "equipo_visitante": "DR Congo",               "sede": "Atlanta",         "fase": "Dieciseisavos", "fecha": "2026-07-01", "hora_espana": "18:00"},
    {"game_id": "53452555", "equipo_local": "Belgium",                 "equipo_visitante": "Senegal",                "sede": "Seattle",         "fase": "Dieciseisavos", "fecha": "2026-07-01", "hora_espana": "22:00"},
    {"game_id": "53452553", "equipo_local": "United States",           "equipo_visitante": "Bosnia and Herzegovina", "sede": "San Francisco",   "fase": "Dieciseisavos", "fecha": "2026-07-02", "hora_espana": "02:00"},
    {"game_id": "53452551", "equipo_local": "Spain",                   "equipo_visitante": "Austria",                "sede": "Los Angeles",     "fase": "Dieciseisavos", "fecha": "2026-07-02", "hora_espana": "21:00"},
    {"game_id": "53452549", "equipo_local": "Portugal",                "equipo_visitante": "Croatia",                "sede": "Toronto",         "fase": "Dieciseisavos", "fecha": "2026-07-03", "hora_espana": "01:00"},
    {"game_id": "53452505", "equipo_local": "Switzerland",             "equipo_visitante": "Algeria",                "sede": "Vancouver",       "fase": "Dieciseisavos", "fecha": "2026-07-03", "hora_espana": "05:00"},
    {"game_id": "53452503", "equipo_local": "Australia",               "equipo_visitante": "Egypt",                  "sede": "Dallas",          "fase": "Dieciseisavos", "fecha": "2026-07-03", "hora_espana": "20:00"},
    {"game_id": "53452569", "equipo_local": "Argentina",               "equipo_visitante": "Cape Verde",             "sede": "Miami",           "fase": "Dieciseisavos", "fecha": "2026-07-04", "hora_espana": "00:00"},
    {"game_id": "53452507", "equipo_local": "Colombia",                "equipo_visitante": "Ghana",                  "sede": "Kansas City",     "fase": "Dieciseisavos", "fecha": "2026-07-04", "hora_espana": "03:30"},
    # Cuartos (fechas aproximadas — se actualizarán vía API cuando se confirmen)
    {"game_id": "QF1",      "equipo_local": "TBD",                    "equipo_visitante": "TBD",                    "sede": "Dallas",          "fase": "Cuartos",       "fecha": "2026-07-09", "hora_espana": ""},
    {"game_id": "QF2",      "equipo_local": "TBD",                    "equipo_visitante": "TBD",                    "sede": "Los Angeles",     "fase": "Cuartos",       "fecha": "2026-07-10", "hora_espana": ""},
    {"game_id": "QF3",      "equipo_local": "TBD",                    "equipo_visitante": "TBD",                    "sede": "Nueva York",      "fase": "Cuartos",       "fecha": "2026-07-11", "hora_espana": ""},
    {"game_id": "QF4",      "equipo_local": "TBD",                    "equipo_visitante": "TBD",                    "sede": "Houston",         "fase": "Cuartos",       "fecha": "2026-07-12", "hora_espana": ""},
    # Semis
    {"game_id": "SF1",      "equipo_local": "TBD",                    "equipo_visitante": "TBD",                    "sede": "Dallas",          "fase": "Semis",         "fecha": "2026-07-16", "hora_espana": ""},
    {"game_id": "SF2",      "equipo_local": "TBD",                    "equipo_visitante": "TBD",                    "sede": "Nueva York",      "fase": "Semis",         "fecha": "2026-07-17", "hora_espana": ""},
    # Tercer puesto + Final
    {"game_id": "TP1",      "equipo_local": "TBD",                    "equipo_visitante": "TBD",                    "sede": "Miami",           "fase": "Tercer_puesto", "fecha": "2026-07-19", "hora_espana": ""},
    {"game_id": "FIN",      "equipo_local": "TBD",                    "equipo_visitante": "TBD",                    "sede": "Nueva York",      "fase": "Final",         "fecha": "2026-07-19", "hora_espana": ""},
]

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _spain_hour(utc_str: str) -> str:
    """Convierte una cadena UTC ISO8601 a hora España (CEST en verano = UTC+2)."""
    if not utc_str:
        return ""
    try:
        dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
        offset = timedelta(hours=SPAIN_UTC_OFFSET_SUMMER)
        local = dt.astimezone(timezone.utc) + offset
        return local.strftime("%H:%M")
    except Exception:
        return ""


def _norm_team(name: str) -> str:
    name = str(name or "").strip()
    return TEAM_NAME_MAP.get(name, name)


def _norm_phase(stage: str) -> str:
    stage = str(stage or "").strip()
    return PHASE_MAP.get(stage, stage)


def _days_rest(team: str, df_played: pd.DataFrame, match_date: str) -> int:
    """Calcula días de descanso mirando el último partido jugado del equipo."""
    if df_played.empty or not team or not match_date:
        return 7
    mask = (df_played["equipo_local"] == team) | (df_played["equipo_visitante"] == team)
    prev = df_played[mask].copy()
    if prev.empty:
        return 7
    prev["fecha"] = pd.to_datetime(prev["fecha"], errors="coerce")
    prev = prev.dropna(subset=["fecha"]).sort_values("fecha")
    target = pd.to_datetime(match_date, errors="coerce")
    if pd.isna(target):
        return 7
    antes = prev[prev["fecha"] < target]
    if antes.empty:
        return 7
    diff = (target - antes.iloc[-1]["fecha"]).days
    return max(1, min(int(diff), 30))


def _cache_is_fresh() -> bool:
    if not CACHE_FILE.exists():
        return False
    age = time.time() - CACHE_FILE.stat().st_mtime
    return age < CACHE_MAX_AGE_HOURS * 3600


def _output_is_fresh() -> bool:
    if not OUTPUT_FILE.exists():
        return False
    age = time.time() - OUTPUT_FILE.stat().st_mtime
    return age < CACHE_MAX_AGE_HOURS * 3600


# ──────────────────────────────────────────────────────────────────────────────
# FUENTE 1: API-Sports / API-Football (la misma que usa tu otro bot)
# ──────────────────────────────────────────────────────────────────────────────

APISPORTS_BASE = "https://v3.football.api-sports.io"

def _apisports_headers() -> dict:
    return {"x-apisports-key": APISPORTS_TOKEN}


def _apisports_fetch(endpoint: str, params: dict | None = None) -> dict | None:
    url = f"{APISPORTS_BASE}/{endpoint}"
    try:
        r = requests.get(url, headers=_apisports_headers(), params=params or {}, timeout=15)
        if r.status_code != 200:
            print(f"  [api-sports] HTTP {r.status_code} en {endpoint}")
            return None
        data = r.json()
        if data.get("errors"):
            print(f"  [api-sports] Error de la API: {data['errors']}")
            return None
        return data
    except Exception as e:
        print(f"  [api-sports] Error: {e}")
        return None


def fetch_from_apisports(days_ahead: int = 14) -> List[Dict[str, Any]]:
    """
    Descarga partidos del Mundial 2026 desde API-Sports (v3.football.api-sports.io).
    Devuelve lista de dicts con las columnas del bot, o [] si falla.
    """
    if not APISPORTS_TOKEN:
        print("  [api-sports] Sin token — salta esta fuente.")
        print("  → Configura: set FOOTBALL_API_KEY=tu_token  (Windows)")
        print("              export FOOTBALL_API_KEY=tu_token (Linux/Mac)")
        return []

    print(f"  [api-sports] Descargando World Cup (league={WORLDCUP_LEAGUE_ID}, season={WORLDCUP_SEASON})...")
    data = _apisports_fetch(
        "fixtures",
        params={"league": WORLDCUP_LEAGUE_ID, "season": WORLDCUP_SEASON},
    )

    if not data or not data.get("response"):
        return []

    raw_matches = data["response"]
    print(f"    → {len(raw_matches)} partidos encontrados")

    # Guardar raw en caché
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(raw_matches, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for m in raw_matches:
        fixture = m.get("fixture", {})
        teams = m.get("teams", {})
        league = m.get("league", {})

        home = _norm_team(teams.get("home", {}).get("name", "TBD"))
        away = _norm_team(teams.get("away", {}).get("name", "TBD"))
        utc_date = fixture.get("date", "")
        fecha = utc_date[:10] if utc_date else ""
        hora_esp = _spain_hour(utc_date)
        stage = league.get("round", "") or ""
        fase = _norm_phase(stage)
        venue = (fixture.get("venue", {}) or {}).get("name", "") or ""
        game_id = str(fixture.get("id", ""))

        rows.append({
            "game_id": game_id,
            "equipo_local": home,
            "equipo_visitante": away,
            "sede": venue,
            "fase": fase,
            "fecha": fecha,
            "hora_espana": hora_esp,
        })

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# FUENTE 0 (prioritaria): ESPN hidden API — gratis, sin token, sin restricción
# de temporada. Es la misma fuente que usa tu otro bot (autobot900...) en
# _espn_scrape() para ligas de clubes; aquí usamos el slug fifa.world.
# ──────────────────────────────────────────────────────────────────────────────

ESPN_WORLDCUP_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"

ESPN_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.6367.82 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}


def fetch_from_espn(date_from: str, date_to: str) -> List[Dict[str, Any]]:
    """
    Descarga partidos del Mundial 2026 desde el endpoint público de ESPN.
    date_from / date_to en formato YYYYMMDD. Devuelve [] si falla.
    """
    print(f"  [espn] Descargando World Cup ({date_from} → {date_to})...")
    params = {"limit": 200, "dates": f"{date_from}-{date_to}"}
    try:
        r = requests.get(ESPN_WORLDCUP_URL, headers=ESPN_HEADERS, params=params, timeout=15)
        if r.status_code != 200:
            print(f"  [espn] HTTP {r.status_code}")
            return []
        data = r.json()
    except Exception as e:
        print(f"  [espn] Error: {e}")
        return []

    events = data.get("events", [])
    if not events:
        print("  [espn] Sin eventos en ese rango de fechas.")
        return []

    print(f"    → {len(events)} partidos encontrados")

    rows = []
    for event in events:
        comp = (event.get("competitions") or [{}])[0]
        competitors = comp.get("competitors", [])
        h = next((c for c in competitors if c.get("homeAway") == "home"), None)
        a = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not h or not a:
            continue

        home = _norm_team(h.get("team", {}).get("displayName", "TBD"))
        away = _norm_team(a.get("team", {}).get("displayName", "TBD"))

        utc_date = event.get("date", "")  # ISO8601, ej. "2026-06-29T19:00Z"
        fecha = utc_date[:10] if utc_date else ""
        hora_esp = _spain_hour(utc_date)

        venue = (comp.get("venue", {}) or {}).get("fullName", "") or ""

        # ESPN trae la fase en notes / season.slug / event.name según el caso
        stage = ""
        notes = comp.get("notes") or []
        if notes:
            stage = notes[0].get("headline", "") or ""
        if not stage:
            stage = (event.get("season", {}) or {}).get("slug", "") or ""
        fase = _norm_phase(stage) or "Grupo"

        game_id = str(event.get("id", ""))

        rows.append({
            "game_id": game_id,
            "equipo_local": home,
            "equipo_visitante": away,
            "sede": venue,
            "fase": fase,
            "fecha": fecha,
            "hora_espana": hora_esp,
        })

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# CÁLCULO DE DÍAS DE DESCANSO
# ──────────────────────────────────────────────────────────────────────────────

def _enrich_rest_days(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Para cada partido en `rows`, calcula los días de descanso de cada equipo
    mirando los partidos anteriores dentro del mismo CSV (todos los del torneo).
    """
    # Construir historial a partir de los propios rows ordenados por fecha
    df_all = pd.DataFrame(rows).copy()
    df_all["fecha"] = pd.to_datetime(df_all["fecha"], errors="coerce")
    df_all = df_all.sort_values("fecha").reset_index(drop=True)

    result = []
    for i, row in df_all.iterrows():
        # Solo partidos anteriores al actual
        df_prev = df_all.iloc[:i].copy()
        rest_l = _days_rest(row["equipo_local"],    df_prev, str(row["fecha"])[:10])
        rest_v = _days_rest(row["equipo_visitante"], df_prev, str(row["fecha"])[:10])
        r = dict(row)
        r["fecha"] = str(row["fecha"])[:10]  # asegurar string YYYY-MM-DD
        r["dias_descanso_local"]      = rest_l
        r["dias_descanso_visitante"]  = rest_v
        result.append(r)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# FILTRADO POR VENTANA TEMPORAL
# ──────────────────────────────────────────────────────────────────────────────

def _filter_days(rows: List[Dict[str, Any]], days_ahead: int) -> List[Dict[str, Any]]:
    today = date.today()
    limit = today + timedelta(days=days_ahead)
    out = []
    for r in rows:
        try:
            d = date.fromisoformat(str(r["fecha"])[:10])
            if today <= d <= limit:
                out.append(r)
        except Exception:
            out.append(r)   # si la fecha es rara, la incluimos igual
    return out


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

FINAL_COLUMNS = [
    "game_id",
    "equipo_local",
    "equipo_visitante",
    "sede",
    "fase",
    "fecha",
    "hora_espana",
    "dias_descanso_local",
    "dias_descanso_visitante",
]


def build_fixtures(days_ahead: int = 14, force: bool = False, all_fixtures: bool = False) -> pd.DataFrame:
    """
    Genera o actualiza telegram_fixtures_worldcup.csv.

    Parámetros:
      days_ahead    — ventana de días a incluir (ignorado si all_fixtures=True)
      force         — ignorar caché y descargar siempre
      all_fixtures  — incluir todos los fixtures del torneo, no solo los próximos
    """
    print(f"[fetch_fixtures] Generando CSV → {OUTPUT_FILE}")

    if not force and _output_is_fresh():
        print(f"  CSV reciente (< {CACHE_MAX_AGE_HOURS}h). Usa --force para recargar.")
        df = pd.read_csv(OUTPUT_FILE)
        print(f"  {len(df)} fixtures ya en CSV.")
        return df

    # ── Fuente 0: ESPN (gratis, sin token, sin restricción de temporada) ───────
    rows: List[Dict[str, Any]] = []
    days_to_fetch = 90 if all_fixtures else max(days_ahead, 30)

    today = date.today()
    date_from = today.strftime("%Y%m%d")
    date_to = (today + timedelta(days=days_to_fetch)).strftime("%Y%m%d")
    # El Mundial 2026 es del 11 jun al 19 jul; si la ventana calculada cae fuera
    # de ese rango, usamos directamente el rango del torneo para no quedarnos sin datos.
    TOURNAMENT_START, TOURNAMENT_END = "20260611", "20260719"
    if date_to < TOURNAMENT_START or date_from > TOURNAMENT_END:
        date_from, date_to = TOURNAMENT_START, TOURNAMENT_END

    print("\n[1/3] Intentando ESPN...")
    espn_rows = fetch_from_espn(date_from, date_to)

    api_rows: List[Dict[str, Any]] = []
    if espn_rows:
        print(f"  ✅ {len(espn_rows)} partidos desde ESPN.")
        rows = espn_rows
        api_rows = espn_rows  # reutiliza la lógica de "viene de API" más abajo
    else:
        print("  ⚠️  ESPN no disponible. Probando api-sports.io...")
        print("\n[2/3] Intentando api-sports.io...")
        api_rows = fetch_from_apisports(days_ahead=days_to_fetch)
        if api_rows:
            print(f"  ✅ {len(api_rows)} partidos desde API.")
            rows = api_rows
        else:
            print("  ⚠️  API no disponible. Usando datos hardcodeados del Mundial 2026.")
            rows = [dict(r) for r in FALLBACK_FIXTURES]

    # ── Enriquecer con días de descanso ──────────────────────────────────────
    print("\n[3/3] Calculando días de descanso...")
    rows = _enrich_rest_days(rows)

    # ── Filtrar por ventana temporal (salvo all_fixtures) ────────────────────
    if not all_fixtures and api_rows:
        # Si vienen de la API, ya vienen filtrados. Si son fallback, filtramos.
        pass
    elif not all_fixtures:
        rows = _filter_days(rows, days_ahead)
        if not rows:
            print(f"  No hay partidos en los próximos {days_ahead} días → usando todos.")
            rows = _enrich_rest_days([dict(r) for r in FALLBACK_FIXTURES])

    # ── Construir DataFrame final ─────────────────────────────────────────────
    df = pd.DataFrame(rows)

    # Añadir columnas que falten con valor por defecto
    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col in ("sede", "hora_espana", "game_id") else 7

    df = df[FINAL_COLUMNS].copy()
    df = df.sort_values("fecha").reset_index(drop=True)

    # ── Guardar ───────────────────────────────────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\n✅ CSV generado: {OUTPUT_FILE}")
    print(f"   {len(df)} partidos | {df['fecha'].min()} → {df['fecha'].max()}")
    if "fase" in df.columns:
        print(f"   Fases: {df['fase'].value_counts().to_dict()}")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Genera telegram_fixtures_worldcup.csv automáticamente"
    )
    parser.add_argument(
        "--days", type=int, default=14,
        help="Días de partidos a incluir (defecto: 14)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ignorar caché y descargar aunque sea reciente"
    )
    parser.add_argument(
        "--all", dest="all_fixtures", action="store_true",
        help="Incluir todos los partidos del torneo (no solo los próximos N días)"
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Mostrar el CSV generado en pantalla"
    )
    args = parser.parse_args()

    df = build_fixtures(days_ahead=args.days, force=args.force, all_fixtures=args.all_fixtures)

    if args.preview:
        print("\nVista previa del CSV:")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()