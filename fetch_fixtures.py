"""
fetch_fixtures.py — Generador automático de telegram_fixtures_worldcup.csv
==========================================================================

Descarga los partidos próximos del Mundial 2026 (u otras competiciones
internacionales) y genera el CSV que lee el bot de Telegram.

FUENTES (en orden de prioridad):
  1. football-data.org API gratuita  →  registro en https://www.football-data.org/client/register
     Solo requiere un token gratuito. Límite: 10 req/min, más que suficiente.
  2. Datos hardcodeados del Mundial 2026 como fallback total.

USO:
  # Una sola vez: registra el token
  export FOOTBALL_DATA_TOKEN="tu_token_aqui"   # Linux/Mac
  set FOOTBALL_DATA_TOKEN=tu_token_aqui         # Windows

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

# Token gratuito de football-data.org  (variable de entorno o edita aquí)
FDORG_TOKEN   = os.environ.get("FOOTBALL_DATA_TOKEN", "")

# IDs de competición en football-data.org:
#   2000 = FIFA World Cup
#   2018 = UEFA Euro
#   2016 = Copa América (cuando está disponible)
COMPETITION_IDS = [2000]

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
# FUENTE 1: football-data.org
# ──────────────────────────────────────────────────────────────────────────────

FDORG_BASE = "https://api.football-data.org/v4"

def _fdorg_headers() -> dict:
    return {"X-Auth-Token": FDORG_TOKEN}


def _fdorg_fetch(endpoint: str, params: dict | None = None) -> dict | None:
    url = f"{FDORG_BASE}/{endpoint}"
    try:
        r = requests.get(url, headers=_fdorg_headers(), params=params or {}, timeout=15)
        if r.status_code == 429:
            print("  [football-data.org] Rate limit — esperando 60s...")
            time.sleep(60)
            r = requests.get(url, headers=_fdorg_headers(), params=params or {}, timeout=15)
        if r.status_code == 200:
            return r.json()
        print(f"  [football-data.org] HTTP {r.status_code} en {endpoint}")
        return None
    except Exception as e:
        print(f"  [football-data.org] Error: {e}")
        return None


def fetch_from_fdorg(days_ahead: int = 14) -> List[Dict[str, Any]]:
    """
    Descarga partidos de las próximas `days_ahead` días de football-data.org.
    Devuelve lista de dicts con las columnas del bot, o [] si falla.
    """
    if not FDORG_TOKEN:
        print("  [football-data.org] Sin token — salta esta fuente.")
        print("  → Regístrate gratis en https://www.football-data.org/client/register")
        print("    y pon: export FOOTBALL_DATA_TOKEN='tu_token'")
        return []

    today     = date.today()
    date_from = today.isoformat()
    date_to   = (today + timedelta(days=days_ahead)).isoformat()

    raw_matches = []
    for comp_id in COMPETITION_IDS:
        print(f"  [football-data.org] Descargando competición {comp_id}...")
        data = _fdorg_fetch(
            f"competitions/{comp_id}/matches",
            params={"dateFrom": date_from, "dateTo": date_to, "status": "SCHEDULED,TIMED"},
        )
        if data and "matches" in data:
            raw_matches.extend(data["matches"])
            print(f"    → {len(data['matches'])} partidos encontrados")
        time.sleep(0.5)   # respetar rate limit

    if not raw_matches:
        return []

    # Guardar raw en caché
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(raw_matches, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for m in raw_matches:
        home = _norm_team(m.get("homeTeam", {}).get("name", "TBD"))
        away = _norm_team(m.get("awayTeam", {}).get("name", "TBD"))
        utc_date = m.get("utcDate", "")
        fecha = utc_date[:10] if utc_date else ""
        hora_esp = _spain_hour(utc_date)
        stage = m.get("stage", "") or m.get("group", "")
        fase = _norm_phase(stage)
        venue = m.get("venue", "") or ""
        game_id = str(m.get("id", ""))

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

    # ── Fuente 1: football-data.org ──────────────────────────────────────────
    rows: List[Dict[str, Any]] = []
    days_to_fetch = 90 if all_fixtures else max(days_ahead, 30)

    print("\n[1/2] Intentando football-data.org...")
    api_rows = fetch_from_fdorg(days_ahead=days_to_fetch)
    if api_rows:
        print(f"  ✅ {len(api_rows)} partidos desde API.")
        rows = api_rows
    else:
        print("  ⚠️  API no disponible. Usando datos hardcodeados del Mundial 2026.")
        rows = [dict(r) for r in FALLBACK_FIXTURES]

    # ── Enriquecer con días de descanso ──────────────────────────────────────
    print("\n[2/2] Calculando días de descanso...")
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
