"""
╔══════════════════════════════════════════════════════════════════════╗
║   ScoutBet V11.0 — HÍBRIDO CUANT + IA, PERSISTENCIA REAL Y MEDIBLE   ║
╠══════════════════════════════════════════════════════════════════════╣
║  NOVEDADES V11 (sobre V10):                                          ║
║   🔐 Credenciales SOLO por entorno / .env (nada hardcodeado)         ║
║   🗄  SQLite (WAL): users, predictions, results — adiós users_v8.json ║
║   📐 Salidas del LLM validadas contra esquema (tipo+rango) + retry   ║
║   📊 Módulo cuantitativo Poisson/Dixon-Coles (puro Python):          ║
║      xG por equipo desde datos reales → P(1X2), P(O2.5), P(BTTS)     ║
║      El LLM ya NO inventa cifras: las interpreta (híbrido)           ║
║   🔁 Self-consistency: Ronda 1 se ejecuta N veces → mediana/moda,    ║
║      alta varianza ⇒ baja la confianza automáticamente               ║
║   ✅ Verificación determinista (CoVe por código): afirmaciones       ║
║      numéricas no soportadas por datos se corrigen/eliminan          ║
║   ⚡ Circuit breaker real por key (closed→open→half-open) +          ║
║      token-bucket por key (sin lock global que serialice todo)      ║
║   📈 Backtesting: cada señal se registra; /calibracion → Brier      ║
║      score y % acierto por mercado con resultados reales             ║
║   📝 Logging estructurado JSON opcional (LOG_JSON=1) + métricas      ║
║   ⚠️  Aviso de juego responsable en cada señal                       ║
║  Pendiente por diseño (paso 6 del plan V11): workers + cola Redis.  ║
║  En Termux/single-process se mantiene JobQueue deliberadamente.      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import logging
import math
import os
import re
import json
import sqlite3
import statistics
import threading
import time
from typing import Optional
from datetime import datetime, timezone, timedelta, time as dt_time

import requests
from groq import Groq
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import (
    Application, CallbackQueryHandler, CommandHandler,
    ContextTypes, MessageHandler, filters, JobQueue
)

# ═══════════════════════════════════════════════════════════
#  CONFIGURACIÓN — SOLO variables de entorno / .env
#  (V11: cero credenciales en el código fuente)
# ═══════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_dotenv(path: str = None) -> None:
    """Carga un .env sin dependencias externas (Termux-friendly).
    No sobreescribe variables ya presentes en el entorno."""
    path = path or os.path.join(BASE_DIR, ".env")
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception as e:
        print(f"[WARN] No se pudo leer .env: {e}")

_load_dotenv()

def _require_env(name: str) -> str:
    val = os.getenv(name, "").strip()
    if not val:
        raise RuntimeError(
            f"Falta la variable de entorno {name}. "
            f"Crea un archivo .env junto al script (mira .env.example) o exporta la variable."
        )
    return val

TELEGRAM_BOT_TOKEN = _require_env("TELEGRAM_BOT_TOKEN")
FOOTBALL_API_KEY   = _require_env("FOOTBALL_API_KEY")
GROQ_MODEL         = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── API Keys de Groq: GROQ_API_KEY_1..N en .env ────────────────────────────
GROQ_API_KEYS: list[str] = []
for _i in range(1, 13):
    _k = os.getenv(f"GROQ_API_KEY_{_i}", "").strip()
    if _k and _k.startswith("gsk_"):
        GROQ_API_KEYS.append(_k)
if not GROQ_API_KEYS:
    raise RuntimeError(
        "Falta al menos una GROQ_API_KEY_1 válida en el entorno/.env "
        "(formato: GROQ_API_KEY_1=gsk_...)"
    )

GROQ_API_KEY = GROQ_API_KEYS[0]

FOOTBALL_API_BASE = "https://v3.football.api-sports.io"
TELEGRAM_LIMIT    = 4000
USERS_FILE        = os.path.join(BASE_DIR, "users_v8.json")     # solo para migración
DB_FILE           = os.path.join(BASE_DIR, os.getenv("DB_FILE", "scoutbet.db"))

# ── Parámetros V11 configurables ────────────────────────────────────────────
SELF_CONSISTENCY_N = max(1, int(os.getenv("SELF_CONSISTENCY_N", "3")))   # pasadas de Ronda 1
LOG_JSON           = os.getenv("LOG_JSON", "0") == "1"                   # logging estructurado

DISCLAIMER = (
    "\n\n⚠️ _Estas señales son estimaciones probabilísticas generadas por un modelo, "
    "NO garantías ni consejo financiero. Apuesta solo lo que puedas permitirte perder. "
    "Juego responsable: jugarbien.es / +18._"
)

# ═══════════════════════════════════════════════════════════
#  CONTROL GROQ — V11: circuit breaker + token bucket POR KEY
#  (sustituye el lock global que serializaba todo el bot)
# ═══════════════════════════════════════════════════════════
GROQ_MIN_INTERVAL = float(os.getenv("GROQ_MIN_INTERVAL", "6.0"))  # intervalo mínimo POR KEY
_groq_paused_until: float = 0.0          # min() de pausas — lo lee job_phase2
API_FOOTBALL_EXHAUSTED = False
_groq_consecutive_errors: int = 0
_groq_inflight: int = 0                  # llamadas en curso (diagnóstico)

# ═══════════════════════════════════════════════════════════
#  MÉTRICAS (observabilidad V11 §2.7)
# ═══════════════════════════════════════════════════════════
METRICS: dict = {
    "groq_calls_ok": 0, "groq_calls_error": 0, "groq_rate_limited": 0,
    "groq_json_invalid": 0, "groq_schema_rejected": 0,
    "breaker_opened": 0, "espn_fallback_used": 0,
    "cove_correcciones": 0, "selfconsistency_baja_confianza": 0,
    "predicciones_registradas": 0, "resultados_registrados": 0,
}
_metrics_lock = threading.Lock()

def metric_inc(name: str, n: int = 1) -> None:
    with _metrics_lock:
        METRICS[name] = METRICS.get(name, 0) + n

# ═══════════════════════════════════════════════════════════
#  ESTADO DEL ESCANER
# ═══════════════════════════════════════════════════════════
_scanner_queue:    list  = []
_scanner_running:  bool  = False
_scanner_found:    int   = 0
_scanner_analyzed: int   = 0
_scanner_total:    int   = 0
SCANNER_INTERVAL = 0.1
SCANNER_MAX_MATCHES = 200
TARGET_MATCHES: dict = {}

# ═══════════════════════════════════════════════════════════
#  DICCIONARIOS
# ═══════════════════════════════════════════════════════════
ALL_LEAGUES = {
    2: "Champions League", 3: "Europa League", 848: "Conference League",
    13: "Copa Libertadores", 14: "Copa Sudamericana",
    39: "Premier League", 140: "La Liga", 135: "Serie A", 78: "Bundesliga", 61: "Ligue 1",
    40: "Championship", 141: "LaLiga 2", 136: "Serie B", 79: "2. Bundesliga",
    94: "Primeira Liga", 88: "Eredivisie", 71: "Serie A (Brasil)",
    128: "Liga Profesional (Arg)", 265: "Primera Div (Col)", 253: "MLS", 169: "Liga MX"
}

ESPN_LEAGUES = {
    "eng.1": "Premier League", "esp.1": "La Liga", "ita.1": "Serie A",
    "ger.1": "Bundesliga", "fra.1": "Ligue 1", "uefa.champions": "Champions League",
    "uefa.europa": "Europa League", "arg.1": "Liga Profesional (Arg)",
    "bra.1": "Serie A (Brasil)", "mex.1": "Liga MX", "usa.1": "MLS",
    "conmebol.libertadores": "Copa Libertadores"
}

APIFOOTBALL_ID_TO_ESPN = {
    2: "uefa.champions", 3: "uefa.europa", 848: "uefa.europa.conference",
    39: "eng.1", 140: "esp.1", 135: "ita.1", 78: "ger.1", 61: "fra.1",
    13: "conmebol.libertadores", 253: "usa.1", 169: "mex.1",
    128: "arg.1", 71: "bra.1", 94: "por.1", 88: "ned.1",
}

VIP_KEYWORDS = {
    "champions", "europa", "conference",
    "premier league",
    "la liga", "serie a", "bundesliga", "ligue 1",
    "laliga", "championship",
    "eredivisie", "primeira liga",
    "libertadores", "sudamericana",
    "mls", "liga mx", "liga profesional",
    "serie a (brasil)",
}

FAKE_PREMIER_LEAGUES = {
    "uganda", "zambia", "zimbabwe", "kenya", "nigeria", "ghana", "tanzania",
    "ethiopia", "rwanda", "malawi", "botswana", "namibia", "south africa",
    "egypt", "algeria", "morocco", "tunisia", "cameroon", "senegal",
    "myanmar", "cambodia", "maldives", "bangladesh", "nepal",
    "solomon", "fiji", "vanuatu", "samoa",
}

ACTIVE_STATUSES   = {"NS", "TBD"}
LIVE_STATUSES     = {"1H", "HT", "2H", "ET", "BT", "P"}
FINISHED_STATUSES = {"FT", "AET", "PEN"}

PAGE_SIZE = 20

# ═══════════════════════════════════════════════════════════
#  MÓDULO DE CONTEXTO COMPETICIONAL — V10
# ═══════════════════════════════════════════════════════════

LIGAS_PERMISIVAS = {
    "champions league", "champions", "uefa champions",
    "copa libertadores", "libertadores",
    "mls",
}

LIGAS_ESTRICTAS = {
    "championship", "serie b", "2. bundesliga", "laliga 2",
    "liga profesional", "serie a (brasil)",
    "primeira liga",
}

FASES_TORNEO_KEYWORDS = {
    "semifinal":    {"tarjetas_factor": -0.8, "goles_factor": 0.0,  "corners_factor": 0.0,  "label": "Semifinal eliminatoria"},
    "final":        {"tarjetas_factor": -1.0, "goles_factor": -0.2, "corners_factor": -0.5, "label": "Final del torneo"},
    "cuartos":      {"tarjetas_factor": -0.5, "goles_factor": 0.0,  "corners_factor": 0.0,  "label": "Cuartos de final"},
    "quarterfinal": {"tarjetas_factor": -0.5, "goles_factor": 0.0,  "corners_factor": 0.0,  "label": "Cuartos de final"},
    "quarter-final":{"tarjetas_factor": -0.5, "goles_factor": 0.0,  "corners_factor": 0.0,  "label": "Cuartos de final"},
    "semi-final":   {"tarjetas_factor": -0.8, "goles_factor": 0.0,  "corners_factor": 0.0,  "label": "Semifinal eliminatoria"},
    "semis":        {"tarjetas_factor": -0.8, "goles_factor": 0.0,  "corners_factor": 0.0,  "label": "Semifinal eliminatoria"},
    "round of 16":  {"tarjetas_factor": -0.3, "goles_factor": 0.0,  "corners_factor": 0.0,  "label": "Octavos de final"},
    "octavos":      {"tarjetas_factor": -0.3, "goles_factor": 0.0,  "corners_factor": 0.0,  "label": "Octavos de final"},
    "copa del rey": {"tarjetas_factor": +0.3, "goles_factor": 0.0,  "corners_factor": 0.0,  "label": "Copa doméstica"},
    "fa cup":       {"tarjetas_factor": +0.4, "goles_factor": 0.0,  "corners_factor": 0.0,  "label": "Copa doméstica"},
    "dfb-pokal":    {"tarjetas_factor": +0.3, "goles_factor": 0.0,  "corners_factor": 0.0,  "label": "Copa doméstica"},
    "coppa italia": {"tarjetas_factor": +0.3, "goles_factor": 0.0,  "corners_factor": 0.0,  "label": "Copa doméstica"},
    "vuelta":       {"tarjetas_factor": +0.4, "goles_factor": +0.3, "corners_factor": +0.5, "label": "Partido de vuelta"},
    "second leg":   {"tarjetas_factor": +0.4, "goles_factor": +0.3, "corners_factor": +0.5, "label": "Partido de vuelta"},
    "return leg":   {"tarjetas_factor": +0.4, "goles_factor": +0.3, "corners_factor": +0.5, "label": "Partido de vuelta"},
}

LIGA_TARJETAS_FACTOR: dict = {
    "champions league":   -0.6,
    "europa league":      -0.2,
    "conference league":   0.0,
    "copa libertadores":  -0.3,
    "copa sudamericana":  +0.2,
    "premier league":     -0.1,
    "la liga":             0.0,
    "laliga":              0.0,
    "serie a":            +0.1,
    "bundesliga":         -0.2,
    "ligue 1":            +0.1,
    "championship":       +0.6,
    "laliga 2":           +0.4,
    "serie b":            +0.5,
    "2. bundesliga":      +0.3,
    "primera liga":        0.0,
    "eredivisie":         -0.1,
    "mls":                -0.2,
    "liga mx":            +0.3,
    "liga profesional":   +0.4,
    "serie a (brasil)":   +0.3,
}


def get_competition_context(league: str, round_info: str = "") -> dict:
    league_l = league.lower().strip()
    round_l  = (round_info or "").lower().strip()
    combined = f"{league_l} {round_l}"

    tarjetas_ajuste = 0.0
    for liga_key, factor in LIGA_TARJETAS_FACTOR.items():
        if liga_key in combined:
            tarjetas_ajuste = factor
            break

    goles_ajuste   = 0.0
    corners_ajuste = 0.0
    fase_detectada = None
    fase_factor    = {}

    for keyword, fdata in FASES_TORNEO_KEYWORDS.items():
        if keyword in combined:
            fase_detectada = fdata["label"]
            fase_factor    = fdata
            tarjetas_ajuste += fdata["tarjetas_factor"]
            goles_ajuste    += fdata["goles_factor"]
            corners_ajuste  += fdata["corners_factor"]
            break

    es_eliminatoria = fase_detectada is not None and "vuelta" not in (fase_detectada or "").lower()
    liga_permisiva  = any(k in league_l for k in LIGAS_PERMISIVAS)
    liga_estricta   = any(k in league_l for k in LIGAS_ESTRICTAS)

    avisos = []

    if tarjetas_ajuste <= -1.0:
        nivel_alerta = "CRITICO"
        avisos.append(
            f"🚨 ALERTA CRÍTICA — TARJETAS: Este partido es una {fase_detectada or 'fase eliminatoria'} "
            f"de la {league}. Los árbitros UEFA/FIFA aplican criterios especialmente permisivos en estas "
            f"fases. Los jugadores EVITAN ACTIVAMENTE las tarjetas para no arriesgar suspensión en la "
            f"siguiente eliminatoria. La media histórica de tarjetas en semifinales/finales de Champions "
            f"es de 2-3, muy por debajo de partidos de liga. NUNCA recomiendes 'Más de 3.5 tarjetas' "
            f"en este contexto sin respaldo estadístico muy sólido del árbitro concreto."
        )
    elif tarjetas_ajuste <= -0.5:
        nivel_alerta = "ALTO"
        avisos.append(
            f"⚠️ AVISO TARJETAS: Fase eliminatoria ({fase_detectada or league}). "
            f"Se esperan MENOS tarjetas de lo habitual. Ajuste aplicado: {tarjetas_ajuste:+.1f} respecto "
            f"a la media del árbitro. Los jugadores son más cautelosos en partidos con consecuencias "
            f"sobre suspensiones. Reduce el umbral esperado de tarjetas en ~{abs(tarjetas_ajuste):.0f}."
        )
    elif tarjetas_ajuste >= 0.5:
        nivel_alerta = "ALTO"
        avisos.append(
            f"⚠️ AVISO TARJETAS: {fase_detectada or league} — "
            f"Se esperan MÁS tarjetas de lo habitual. Ajuste: {tarjetas_ajuste:+.1f}."
        )
    else:
        nivel_alerta = "NORMAL"

    if liga_permisiva and not fase_detectada:
        avisos.append(
            f"📋 CONTEXTO LIGA: La {league} usa árbitros internacionales con estilo permisivo. "
            f"Las medias de tarjetas son estructuralmente más bajas que en ligas domésticas."
        )

    if goles_ajuste != 0.0:
        avisos.append(
            f"📋 CONTEXTO GOLES: Ajuste por fase ({fase_detectada}): {goles_ajuste:+.1f} goles respecto a la media habitual."
        )

    if corners_ajuste != 0.0:
        avisos.append(
            f"📋 CONTEXTO CORNERS: Ajuste por fase ({fase_detectada}): {corners_ajuste:+.1f} corners respecto a la media habitual."
        )

    aviso_prompt = "\n".join(avisos) if avisos else ""

    logger.info(
        f"CompetitionContext [{league}|{round_info}]: "
        f"tarjetas={tarjetas_ajuste:+.1f}, goles={goles_ajuste:+.1f}, "
        f"corners={corners_ajuste:+.1f}, fase={fase_detectada}, alerta={nivel_alerta}"
    )

    return {
        "tarjetas_ajuste":  tarjetas_ajuste,
        "goles_ajuste":     goles_ajuste,
        "corners_ajuste":   corners_ajuste,
        "es_eliminatoria":  es_eliminatoria,
        "fase_detectada":   fase_detectada,
        "liga_permisiva":   liga_permisiva,
        "liga_estricta":    liga_estricta,
        "aviso_prompt":     aviso_prompt,
        "nivel_alerta":     nivel_alerta,
        "league":           league,
        "round_info":       round_info,
    }

# ═══════════════════════════════════════════════════════════
#  LOGGING & TIEMPO
# ═══════════════════════════════════════════════════════════
class _JsonLogFormatter(logging.Formatter):
    """Logging estructurado (V11 §2.7): una línea JSON por evento, fácil de
    ingerir en Loki/CloudWatch/grep+jq. Actívalo con LOG_JSON=1."""
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

if LOG_JSON:
    _h = logging.StreamHandler()
    _h.setFormatter(_JsonLogFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[_h])
else:
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", level=logging.INFO)
logger = logging.getLogger("ScoutBetV11")

def _spain_offset() -> timedelta:
    now_utc   = datetime.now(timezone.utc)
    year      = now_utc.year
    mar31     = datetime(year, 3, 31, 1, 0, tzinfo=timezone.utc)
    dst_start = mar31 - timedelta(days=(mar31.weekday() + 1) % 7)
    oct31     = datetime(year, 10, 31, 1, 0, tzinfo=timezone.utc)
    dst_end   = oct31 - timedelta(days=(oct31.weekday() + 1) % 7)
    return timedelta(hours=2) if dst_start <= now_utc < dst_end else timedelta(hours=1)

def now_spain() -> datetime:
    return datetime.now(timezone.utc) + _spain_offset()

def utc_for_spain_hour(h: int) -> int:
    return (h - int(_spain_offset().total_seconds() // 3600)) % 24

def parse_fixture_time(iso_str: str) -> str:
    try:
        if not iso_str: return "?"
        s = iso_str.replace("Z", "+00:00")
        utc_dt = datetime.fromisoformat(s)
        if utc_dt.tzinfo is None:
            utc_dt = utc_dt.replace(tzinfo=timezone.utc)
        spain_dt   = utc_dt + _spain_offset()
        today_es   = now_spain().date()
        match_date = spain_dt.date()
        time_str   = spain_dt.strftime("%H:%M")
        if match_date == today_es:
            return time_str
        elif match_date > today_es:
            return f"MAÑ {time_str}"
        else:
            return f"AYR {time_str}"
    except Exception:
        return iso_str[11:16] if iso_str and len(iso_str) > 15 else "?"

def get_minutes_to_kickoff(iso_str: str) -> int:
    try:
        if not iso_str: return 9999
        s = iso_str.replace("Z", "+00:00")
        utc_dt = datetime.fromisoformat(s)
        if utc_dt.tzinfo is None: utc_dt = utc_dt.replace(tzinfo=timezone.utc)
        diff = utc_dt - datetime.now(timezone.utc)
        return int(diff.total_seconds() / 60)
    except Exception:
        return 9999

# ═══════════════════════════════════════════════════════════
#  PERSISTENCIA — SQLite (V11 §2.5)
#  Sustituye users_v8.json: transaccional, sin race conditions,
#  y añade predictions/results para backtesting y calibración.
# ═══════════════════════════════════════════════════════════
_db_lock = threading.Lock()

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn

def db_init() -> None:
    with _db_lock, _db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            chat_id            TEXT PRIMARY KEY,
            interval           INTEGER NOT NULL DEFAULT 30,
            last_alert         REAL    NOT NULL DEFAULT 0,
            active             INTEGER NOT NULL DEFAULT 1,
            notified_matches   TEXT    NOT NULL DEFAULT '[]',
            created_at         TEXT    DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            fixture_id   INTEGER NOT NULL,
            home         TEXT, away TEXT, league TEXT,
            kickoff_utc  TEXT,
            market       TEXT NOT NULL,          -- '1X2', 'over_2_5', 'btts', ...
            prediccion   TEXT NOT NULL,          -- 'home'/'draw'/'away', 'si'/'no'...
            probabilidad REAL,                   -- prob del modelo cuantitativo [0,1]
            confianza    TEXT,                   -- alta/media/baja (pipeline)
            ronda_json   TEXT,                   -- trazabilidad: patron+quant serializados
            created_at   TEXT DEFAULT (datetime('now')),
            UNIQUE(fixture_id, market)           -- idempotencia
        );
        CREATE TABLE IF NOT EXISTS results (
            fixture_id   INTEGER PRIMARY KEY,
            goles_local  INTEGER, goles_visitante INTEGER,
            resultado    TEXT,                   -- 'home'/'draw'/'away'
            btts         INTEGER,                -- 0/1
            over_2_5     INTEGER,                -- 0/1
            fetched_at   TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_pred_fixture ON predictions(fixture_id);
        """)
    _migrate_users_json()

def _migrate_users_json() -> None:
    """Migración única desde users_v8.json → tabla users."""
    if not os.path.exists(USERS_FILE):
        return
    try:
        with open(USERS_FILE) as f:
            legacy = json.load(f)
        with _db_lock, _db() as c:
            for cid, prefs in legacy.items():
                c.execute(
                    "INSERT OR IGNORE INTO users(chat_id, interval, last_alert, active, notified_matches) "
                    "VALUES (?,?,?,?,?)",
                    (str(cid), int(prefs.get("interval", 30)), float(prefs.get("last_alert", 0)),
                     1 if prefs.get("active", True) else 0,
                     json.dumps(prefs.get("notified_matches", []))),
                )
        os.rename(USERS_FILE, USERS_FILE + ".migrated")
        logger.info(f"Migrados {len(legacy)} usuarios de JSON a SQLite ({DB_FILE})")
    except Exception as e:
        logger.error(f"Migración users_v8.json fallida: {e}")

def load_users() -> dict:
    users = {}
    try:
        with _db_lock, _db() as c:
            for r in c.execute("SELECT * FROM users"):
                users[r["chat_id"]] = {
                    "interval": r["interval"], "last_alert": r["last_alert"],
                    "active": bool(r["active"]),
                    "notified_matches": json.loads(r["notified_matches"] or "[]"),
                }
    except Exception as e:
        logger.error(f"load_users: {e}")
    return users

def save_users(users: dict) -> None:
    try:
        with _db_lock, _db() as c:
            for cid, prefs in users.items():
                c.execute(
                    "INSERT INTO users(chat_id, interval, last_alert, active, notified_matches) "
                    "VALUES (?,?,?,?,?) "
                    "ON CONFLICT(chat_id) DO UPDATE SET interval=excluded.interval, "
                    "last_alert=excluded.last_alert, active=excluded.active, "
                    "notified_matches=excluded.notified_matches",
                    (str(cid), int(prefs.get("interval", 30)), float(prefs.get("last_alert", 0)),
                     1 if prefs.get("active", True) else 0,
                     json.dumps(prefs.get("notified_matches", []))),
                )
    except Exception as e:
        logger.error(f"save_users: {e}")

def db_log_prediction(fixture_id: int, home: str, away: str, league: str, kickoff_utc: str,
                      market: str, prediccion: str, probabilidad: Optional[float],
                      confianza: str, ronda_json: dict) -> None:
    """Registra cada señal emitida (V11 §2.5). UNIQUE(fixture,market) = idempotente."""
    try:
        with _db_lock, _db() as c:
            c.execute(
                "INSERT OR IGNORE INTO predictions"
                "(fixture_id, home, away, league, kickoff_utc, market, prediccion, probabilidad, confianza, ronda_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (fixture_id, home, away, league, kickoff_utc, market, prediccion,
                 probabilidad, confianza, json.dumps(ronda_json, ensure_ascii=False)[:20000]),
            )
        metric_inc("predicciones_registradas")
    except Exception as e:
        logger.error(f"db_log_prediction: {e}")

def db_pending_results(max_rows: int = 25) -> list:
    """Fixtures con predicción, kickoff pasado hace >3h y sin resultado guardado."""
    try:
        with _db_lock, _db() as c:
            rows = c.execute(
                "SELECT DISTINCT p.fixture_id FROM predictions p "
                "LEFT JOIN results r ON r.fixture_id = p.fixture_id "
                "WHERE r.fixture_id IS NULL AND p.fixture_id > 0 "
                "AND p.kickoff_utc IS NOT NULL AND p.kickoff_utc != '' "
                "AND datetime(p.kickoff_utc) < datetime('now', '-3 hours') "
                "LIMIT ?", (max_rows,)
            ).fetchall()
        return [r["fixture_id"] for r in rows]
    except Exception as e:
        logger.error(f"db_pending_results: {e}")
        return []

def db_save_result(fixture_id: int, gl: int, gv: int) -> None:
    resultado = "home" if gl > gv else ("away" if gv > gl else "draw")
    try:
        with _db_lock, _db() as c:
            c.execute(
                "INSERT OR REPLACE INTO results(fixture_id, goles_local, goles_visitante, resultado, btts, over_2_5) "
                "VALUES (?,?,?,?,?,?)",
                (fixture_id, gl, gv, resultado, 1 if (gl > 0 and gv > 0) else 0, 1 if (gl + gv) > 2 else 0),
            )
        metric_inc("resultados_registrados")
    except Exception as e:
        logger.error(f"db_save_result: {e}")

def db_calibration() -> dict:
    """Brier score y % acierto por mercado (V11 §2.5).
    Brier = media de (probabilidad_predicha − resultado_real)²; 0 = perfecto, 0.25 = azar en binario."""
    out = {}
    try:
        with _db_lock, _db() as c:
            rows = c.execute(
                "SELECT p.market, p.prediccion, p.probabilidad, r.resultado, r.btts, r.over_2_5 "
                "FROM predictions p JOIN results r ON r.fixture_id = p.fixture_id "
                "WHERE p.probabilidad IS NOT NULL"
            ).fetchall()
        for r in rows:
            market = r["market"]
            if market == "1X2":
                hit = 1.0 if r["prediccion"] == r["resultado"] else 0.0
            elif market == "over_2_5":
                hit = 1.0 if (r["prediccion"] == "si") == bool(r["over_2_5"]) else 0.0
            elif market == "btts":
                hit = 1.0 if (r["prediccion"] == "si") == bool(r["btts"]) else 0.0
            else:
                continue
            m = out.setdefault(market, {"n": 0, "aciertos": 0, "brier_sum": 0.0})
            m["n"] += 1
            m["aciertos"] += int(hit)
            m["brier_sum"] += (r["probabilidad"] - hit) ** 2
        for m in out.values():
            m["hit_rate"] = m["aciertos"] / m["n"]
            m["brier"] = m["brier_sum"] / m["n"]
    except Exception as e:
        logger.error(f"db_calibration: {e}")
    return out

db_init()
subscribed_users: dict = load_users()

def ensure_user_active(chat_id: int):
    cid = str(chat_id)
    if cid not in subscribed_users:
        subscribed_users[cid] = {"interval": 30, "last_alert": 0, "active": True, "notified_matches": []}
    else:
        subscribed_users[cid]["active"] = True
    save_users(subscribed_users)

def clear_daily_notifications():
    for uid in subscribed_users:
        subscribed_users[uid]["notified_matches"] = []
    save_users(subscribed_users)

async def notify_users(context: ContextTypes.DEFAULT_TYPE, message: str):
    for uid, prefs in subscribed_users.items():
        if prefs.get("active", True):
            try:
                for chunk in split_message(message):
                    await context.bot.send_message(chat_id=uid, text=chunk, parse_mode="Markdown")
            except Exception: pass

# ═══════════════════════════════════════════════════════════
#  CLIENTE GROQ — V11: CIRCUIT BREAKER + TOKEN BUCKET POR KEY
#  Estados por key: CLOSED (verde) → OPEN (pausada tras fallos)
#  → HALF-OPEN (una llamada de prueba al expirar la pausa).
#  El rate-limit es POR KEY, no global: dos keys distintas pueden
#  llamar en paralelo (necesario para self-consistency §2.3).
# ═══════════════════════════════════════════════════════════
_groq_clients: list[Groq] = [Groq(api_key=k) for k in GROQ_API_KEYS]

_key_paused_until:  list[float] = [0.0]  * len(GROQ_API_KEYS)   # OPEN hasta este instante
_key_consec_errors: list[int]   = [0]    * len(GROQ_API_KEYS)
_key_quota_dead:    list[bool]  = [False]* len(GROQ_API_KEYS)   # OPEN 24h (cuota)
_key_half_open:     list[bool]  = [False]* len(GROQ_API_KEYS)   # probe en curso
_key_last_call:     list[float] = [0.0]  * len(GROQ_API_KEYS)   # token bucket por key
_key_opens_count:   list[int]   = [0]    * len(GROQ_API_KEYS)   # métrica: veces abierto
_key_locks: list[asyncio.Lock] = [asyncio.Lock() for _ in GROQ_API_KEYS]

_groq_key_index: int = 0

def _breaker_state(idx: int) -> str:
    now = time.monotonic()
    if _key_quota_dead[idx]:
        return "OPEN_QUOTA"
    if now < _key_paused_until[idx]:
        return "OPEN"
    if _key_half_open[idx]:
        return "HALF_OPEN"
    return "CLOSED"

def _breaker_trip(idx: int, seconds: float) -> None:
    """CLOSED/HALF_OPEN → OPEN durante `seconds`."""
    global _groq_paused_until
    _key_paused_until[idx] = time.monotonic() + seconds
    _key_half_open[idx] = False
    _key_opens_count[idx] += 1
    metric_inc("breaker_opened")
    _groq_paused_until = min(_key_paused_until)

def _breaker_success(idx: int) -> None:
    """Cualquier estado → CLOSED."""
    _key_consec_errors[idx] = 0
    _key_half_open[idx] = False

def _next_groq_key() -> int:
    """Elige la siguiente key en CLOSED; si la pausa de una expiró, entra en
    HALF_OPEN (se le permite una llamada de prueba). Si todas están abiertas,
    devuelve la que antes libere."""
    global _groq_key_index
    now = time.monotonic()
    n = len(GROQ_API_KEYS)

    for i in range(n):
        idx = (_groq_key_index + i) % n
        if _key_quota_dead[idx]:
            continue
        if now >= _key_paused_until[idx]:
            if _key_consec_errors[idx] > 0 and not _key_half_open[idx]:
                _key_half_open[idx] = True   # transición OPEN → HALF_OPEN
            _groq_key_index = (idx + 1) % n
            return idx

    vivas = [i for i in range(n) if not _key_quota_dead[i]]
    pool  = vivas if vivas else list(range(n))
    best  = min(pool, key=lambda i: _key_paused_until[i])
    _groq_key_index = (best + 1) % n
    return best

async def ask_groq(prompt: str, max_retries: int = 4, expect_json: bool = False, temperature: float = 0.3) -> str:
    """V11: sin lock global. Cada key tiene su propio lock + intervalo mínimo,
    de modo que llamadas con keys distintas pueden ir en paralelo. Cuando se
    espera JSON, se pide con response_format=json_object (salida estructurada
    nativa de Groq §2.1) en vez de confiar solo en el prompt."""
    global _groq_paused_until, _groq_consecutive_errors, _groq_inflight

    logger.debug(f"Groq prompt ~{len(prompt)//4} tokens, temp={temperature}, expect_json={expect_json}")

    for attempt in range(max_retries):
        idx = _next_groq_key()
        now = time.monotonic()

        if now < _key_paused_until[idx]:
            wait = min(_key_paused_until[idx] - now + 0.5, 90)
            logger.info(f"Key#{idx+1} en OPEN, esperando {wait:.0f}s...")
            await asyncio.sleep(wait)

        async with _key_locks[idx]:
            # token bucket por key
            elapsed = time.monotonic() - _key_last_call[idx]
            if elapsed < GROQ_MIN_INTERVAL:
                await asyncio.sleep(GROQ_MIN_INTERVAL - elapsed)

            client = _groq_clients[idx]
            estado = _breaker_state(idx)
            logger.info(f"Groq → Key#{idx+1} [{estado}] (intento {attempt+1}/{max_retries})")

            kwargs = dict(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3200,
                temperature=temperature,
            )
            if expect_json:
                kwargs["response_format"] = {"type": "json_object"}

            _groq_inflight += 1
            try:
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None, lambda: client.chat.completions.create(**kwargs)
                )
                content = (response.choices[0].message.content or "").strip()
                _key_last_call[idx] = time.monotonic()

                if expect_json and not _parse_groq_json(content):
                    metric_inc("groq_json_invalid")
                    if attempt < max_retries - 1:
                        logger.warning(f"JSON inválido en intento {attempt+1}; reintentando...")
                        prompt += ("\n\nIMPORTANTE: Devuelve ÚNICAMENTE un objeto JSON válido, "
                                   "sin markdown ni texto adicional. Usa null donde no tengas certeza.")
                        continue
                    logger.error(f"JSON inválido tras {max_retries} intentos: {content[:200]}...")
                    return ""

                _breaker_success(idx)
                _groq_consecutive_errors = 0
                metric_inc("groq_calls_ok")
                return content

            except Exception as e:
                _key_last_call[idx] = time.monotonic()
                _key_consec_errors[idx] += 1
                _groq_consecutive_errors += 1
                metric_inc("groq_calls_error")
                error_str = str(e).lower()

                if "429" in error_str or "rate_limit" in error_str or "too many" in error_str:
                    metric_inc("groq_rate_limited")
                    _breaker_trip(idx, min(60 * _key_consec_errors[idx], 300))
                    logger.warning(f"Key#{idx+1} rate limit — breaker OPEN. Rotando...")
                elif "quota" in error_str or "insufficient_quota" in error_str:
                    _key_quota_dead[idx] = True
                    _breaker_trip(idx, 86400)
                    logger.error(f"Key#{idx+1} cuota agotada — OPEN 24h.")
                    if all(_key_quota_dead):
                        logger.error("Todas las keys con cuota agotada.")
                        return ""
                elif "503" in error_str or "unavailable" in error_str or "overloaded" in error_str:
                    _breaker_trip(idx, 30 * (attempt + 1))
                    logger.warning(f"Key#{idx+1} sobrecargada — breaker OPEN {30*(attempt+1)}s")
                else:
                    logger.error(f"Error Groq Key#{idx+1}: {e}")
            finally:
                _groq_inflight -= 1

            if attempt < max_retries - 1:
                await asyncio.sleep(min(5 * (2 ** attempt), 30))

    logger.error("Groq: máximo de reintentos alcanzado en todas las keys.")
    return ""

# ═══════════════════════════════════════════════════════════
#  VALIDACIÓN DE ESQUEMA (V11 §2.1)
#  Cada ronda del LLM se valida contra un esquema tipado con rangos.
#  Si no valida → se descartan los campos inválidos y se reintenta.
#  (Puro Python: pydantic pesa demasiado en Termux; mismo efecto.)
# ═══════════════════════════════════════════════════════════
# formato: campo -> (tipos_permitidos, min, max) | (tipos,) para no numéricos
SCHEMA_RONDA1 = {
    "estilo_local":            (str,),
    "estilo_visitante":        (str,),
    "ritmo_esperado":          (str,),
    "goles_esperados_total":   ((int, float), 0.0, 8.0),
    "corners_esperados_total": ((int, float), 0.0, 25.0),
    "corners_local_mayor":     (bool,),
    "corners_confianza":       (str,),
    "senal_corners":           (str,),
    "tarjetas_esperadas":      ((int, float), 0.0, 12.0),
    "partido_abierto":         (bool,),
    "primer_gol_antes_30":     (bool,),
    "btts_probable":           (bool,),
    "penalti_probable":        (bool,),
    "observaciones_clave":     (str,),
    "confianza":               (str,),
}

def _validate_schema(data: dict, schema: dict) -> tuple[dict, list[str]]:
    """Devuelve (data_limpia, errores). Campos con tipo/rango inválido → None."""
    if not isinstance(data, dict):
        return {}, ["respuesta no es un objeto JSON"]
    clean, errores = {}, []
    for campo, spec in schema.items():
        val = data.get(campo)
        if val is None:
            clean[campo] = None
            continue
        tipos = spec[0] if isinstance(spec[0], tuple) else (spec[0],)
        # bool es subclase de int en Python: excluirlo de campos numéricos
        if float in tipos and isinstance(val, bool):
            errores.append(f"{campo}: bool donde se esperaba número")
            clean[campo] = None
            continue
        if not isinstance(val, tipos):
            errores.append(f"{campo}: tipo {type(val).__name__} inválido")
            clean[campo] = None
            continue
        if len(spec) == 3 and isinstance(val, (int, float)):
            lo, hi = spec[1], spec[2]
            if not (lo <= float(val) <= hi):
                errores.append(f"{campo}: {val} fuera de rango [{lo},{hi}]")
                clean[campo] = None
                continue
        clean[campo] = val
    # conservar campos extra no contemplados (p.ej. listas de ronda 2/3)
    for k, v in data.items():
        if k not in clean:
            clean[k] = v
    return clean, errores

async def ask_groq_json(prompt: str, schema: Optional[dict] = None,
                        max_retries: int = 4, temperature: float = 0.3) -> dict:
    """ask_groq + parseo + validación de esquema con reintento dirigido."""
    for intento in range(2):
        resp = await ask_groq(prompt, max_retries=max_retries, expect_json=True, temperature=temperature)
        data = _parse_groq_json(resp)
        if not data:
            return {}
        if not schema:
            return data
        clean, errores = _validate_schema(data, schema)
        if not errores:
            return clean
        metric_inc("groq_schema_rejected")
        logger.warning(f"Esquema rechazado ({len(errores)} campos): {errores[:4]}")
        if intento == 0:
            prompt += ("\n\n⚠️ CORRECCIÓN OBLIGATORIA: tu respuesta anterior tenía estos errores de "
                       f"tipo/rango: {'; '.join(errores[:6])}. Corrígelos. Usa null si no tienes certeza.")
            continue
        return clean  # última pasada: campos inválidos ya en None
    return {}

# ═══════════════════════════════════════════════════════════
#  API FOOTBALL & FALLBACK ESPN
# ═══════════════════════════════════════════════════════════
_api_cache: dict = {}
_api_cache_ts: dict = {}

def _clean_api_cache():
    now_ts = datetime.now().timestamp()
    expired = [k for k, v in _api_cache_ts.items() if now_ts - v > 3600]
    for k in expired:
        _api_cache.pop(k, None)
        _api_cache_ts.pop(k, None)

def football_api(endpoint: str, params: dict, ttl: int = 600) -> Optional[dict]:
    global API_FOOTBALL_EXHAUSTED
    if API_FOOTBALL_EXHAUSTED or not FOOTBALL_API_KEY: return None

    if len(_api_cache) > 100:
        _clean_api_cache()

    key = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
    now_ts = datetime.now().timestamp()
    if key in _api_cache and now_ts - _api_cache_ts.get(key, 0) < ttl:
        return _api_cache[key]
    try:
        resp = requests.get(
            f"{FOOTBALL_API_BASE}/{endpoint}",
            headers={"x-apisports-key": FOOTBALL_API_KEY}, params=params, timeout=10
        )
        data = resp.json()
        if data.get("errors") and "requests" in data.get("errors", {}):
            API_FOOTBALL_EXHAUSTED = True
            logger.error("API-Football AGOTADA.")
            return None
        _api_cache[key] = data
        _api_cache_ts[key] = now_ts
        return data
    except Exception: return None

def _safe_score(competitor: dict) -> int:
    try:
        s = competitor.get("score", "")
        return int(s) if s not in (None, "", "-") else 0
    except Exception: return 0

_fallback_cache = {"live": {"data": [], "ts": 0}, "today": {"data": [], "ts": 0}}

def get_fallback_matches(only_live=False) -> list:
    metric_inc("espn_fallback_used")
    cache_key = "live" if only_live else "today"
    now_ts = datetime.now().timestamp()
    cached = _fallback_cache[cache_key]
    if cached["data"] and now_ts - cached["ts"] < 60: return cached["data"]

    matches = []
    seen_ids = set()

    for league_code, league_name in ESPN_LEAGUES.items():
        try:
            url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league_code}/scoreboard"
            resp = requests.get(url, timeout=8)
            for event in resp.json().get("events", []):
                state = event.get("status", {}).get("type", {}).get("state")
                if only_live and state != "in": continue
                if not only_live and state == "post": continue

                event_id = int(event["id"])
                if event_id in seen_ids: continue
                seen_ids.add(event_id)

                date_str = event.get("date", "")
                mins = get_minutes_to_kickoff(date_str)
                if state == "post":
                    continue
                elif state != "in" and mins > 1440:
                    continue

                comps = event["competitions"][0]["competitors"]
                home = next((c for c in comps if c["homeAway"] == "home"), None)
                away = next((c for c in comps if c["homeAway"] == "away"), None)
                if not home or not away: continue

                matches.append({
                    "fixture_id": event_id,
                    "home": home["team"].get("displayName") or home["team"].get("name", "?"),
                    "away": away["team"].get("displayName") or away["team"].get("name", "?"),
                    "time": "En vivo" if state == "in" else parse_fixture_time(date_str),
                    "date_iso": date_str,
                    "league": league_name, "league_id": 0,
                    "espn_league_code": league_code,
                    "status": "LIVE" if state == "in" else "NS",
                    "is_live": state == "in",
                    "minute": event.get("status", {}).get("type", {}).get("shortDetail", "").split("-")[-1].strip() if state == "in" else None,
                    "mins_to_kickoff": mins,
                    "home_goals": _safe_score(home), "away_goals": _safe_score(away),
                    "referee": None, "venue": None
                })
        except Exception: pass

    if matches:
        _fallback_cache[cache_key] = {"data": sorted(matches, key=lambda x: (not x["is_live"], x["time"])), "ts": now_ts}
    return matches

def get_live_matches() -> list:
    if API_FOOTBALL_EXHAUSTED: return get_fallback_matches(only_live=True)
    today = now_spain().strftime("%Y-%m-%d")
    data = football_api("fixtures", {"date": today}, ttl=120)
    if not data or not data.get("response"): return get_fallback_matches(only_live=True)

    matches = []
    for f in data["response"]:
        if f["fixture"]["status"]["short"] in LIVE_STATUSES:
            matches.append({
                "fixture_id": f["fixture"]["id"], "home": f["teams"]["home"]["name"], "away": f["teams"]["away"]["name"],
                "time": parse_fixture_time(f["fixture"]["date"]), "date_iso": f["fixture"]["date"],
                "league": f["league"]["name"], "league_id": f["league"]["id"],
                "status": f["fixture"]["status"]["short"], "is_live": True,
                "minute": f["fixture"]["status"].get("elapsed"),
                "home_goals": f["goals"].get("home", 0), "away_goals": f["goals"].get("away", 0),
                "referee": f["fixture"].get("referee") or None,
                "venue": f["fixture"].get("venue", {}).get("name") or None,
                "round": f["league"].get("round", "") or "",
            })
    return matches

def get_todays_matches() -> list:
    if API_FOOTBALL_EXHAUSTED:
        return get_fallback_matches(only_live=False)

    now_utc       = datetime.now(timezone.utc)
    window_future = now_utc + timedelta(hours=24)

    dates_to_fetch = {
        now_spain().strftime("%Y-%m-%d"),
        (now_spain() + timedelta(days=1)).strftime("%Y-%m-%d"),
    }

    raw_fixtures: list = []
    for date_str in sorted(dates_to_fetch):
        data = football_api("fixtures", {"date": date_str}, ttl=120)
        if data and data.get("response"):
            raw_fixtures.extend(data["response"])

    if not raw_fixtures:
        return get_fallback_matches(only_live=False)

    matches: list = []
    seen: set = set()

    for f in raw_fixtures:
        fid = f["fixture"]["id"]
        if fid in seen:
            continue
        seen.add(fid)

        status   = f["fixture"]["status"]["short"]
        date_iso = f["fixture"]["date"]

        try:
            s = date_iso.replace("Z", "+00:00")
            match_utc = datetime.fromisoformat(s)
            if match_utc.tzinfo is None:
                match_utc = match_utc.replace(tzinfo=timezone.utc)
        except Exception:
            match_utc = now_utc

        if status in LIVE_STATUSES:
            pass
        elif status in FINISHED_STATUSES:
            continue
        else:
            if match_utc > window_future:
                continue
            if match_utc < now_utc - timedelta(minutes=10):
                continue

        matches.append({
            "fixture_id": fid,
            "home":  f["teams"]["home"]["name"],
            "away":  f["teams"]["away"]["name"],
            "time":  parse_fixture_time(date_iso),
            "date_iso": date_iso,
            "league": f["league"]["name"],
            "league_id": f["league"]["id"],
            "status": status,
            "is_live": status in LIVE_STATUSES,
            "minute": f["fixture"]["status"].get("elapsed"),
            "mins_to_kickoff": get_minutes_to_kickoff(date_iso),
            "home_goals": f["goals"].get("home", 0),
            "away_goals": f["goals"].get("away", 0),
            "referee": f["fixture"].get("referee") or None,
            "venue":   f["fixture"].get("venue", {}).get("name") or None,
            "round":   f["league"].get("round", "") or "",
        })

    def _sort_key(m):
        try:
            s = m["date_iso"].replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return (not m["is_live"], dt)
        except Exception:
            return (not m["is_live"], datetime.max.replace(tzinfo=timezone.utc))
    return sorted(matches, key=_sort_key)

def get_deep_data(fixture_id: int, league_id: int, home_team_id: int = 0, away_team_id: int = 0) -> dict:
    deep_data = {
        "lineups": None, "stats_home": {}, "stats_away": {},
        "h2h": [], "injuries_home": [], "injuries_away": [],
        "home_team_id": home_team_id, "away_team_id": away_team_id,
    }
    if league_id == 0 and fixture_id == 0: return deep_data

    if fixture_id:
        # Lineups: NO cachear si la respuesta viene vacía (equipo aún no publicó)
        _lineup_cache_key = f"fixtures/lineups:{json.dumps({'fixture': fixture_id}, sort_keys=True)}"
        if _lineup_cache_key in _api_cache and not _api_cache[_lineup_cache_key].get("response"):
            del _api_cache[_lineup_cache_key]  # invalidar caché vacía
        l_data = football_api("fixtures/lineups", {"fixture": fixture_id}, ttl=120)  # TTL corto: 2 min
        if l_data and l_data.get("response"):
            try:
                h_line = l_data["response"][0]
                a_line = l_data["response"][1]
                h_players = [p["player"]["name"] for p in h_line.get("startXI", [])]
                a_players = [p["player"]["name"] for p in a_line.get("startXI", [])]
                if h_players and a_players:
                    deep_data["lineups"] = {
                        "home_formation": h_line.get("formation", "?"),
                        "away_formation": a_line.get("formation", "?"),
                        "home_coach": h_line.get("coach", {}).get("name", "?"),
                        "away_coach": a_line.get("coach", {}).get("name", "?"),
                        "home_players": h_players,
                        "away_players": a_players,
                    }
                    if not home_team_id:
                        deep_data["home_team_id"] = h_line.get("team", {}).get("id", 0)
                    if not away_team_id:
                        deep_data["away_team_id"] = a_line.get("team", {}).get("id", 0)
                else:
                    # response existe pero startXI vacío — también invalidar para próxima llamada
                    if _lineup_cache_key in _api_cache:
                        del _api_cache[_lineup_cache_key]
            except Exception: pass
        else:
            # respuesta vacía o None — limpiar caché para que reintente en próxima llamada
            if _lineup_cache_key in _api_cache:
                del _api_cache[_lineup_cache_key]

    h_id = deep_data.get("home_team_id") or home_team_id
    a_id = deep_data.get("away_team_id") or away_team_id
    if h_id and a_id:
        h2h_data = football_api("fixtures/headtohead", {"h2h": f"{h_id}-{a_id}", "last": 5}, ttl=3600)
        if h2h_data and h2h_data.get("response"):
            for f in h2h_data["response"][:5]:
                try:
                    winner = (f["teams"]["home"]["name"] if f["teams"]["home"].get("winner")
                              else (f["teams"]["away"]["name"] if f["teams"]["away"].get("winner") else "Empate"))
                    deep_data["h2h"].append({
                        "date": f["fixture"]["date"][:10],
                        "home": f["teams"]["home"]["name"],
                        "away": f["teams"]["away"]["name"],
                        "score": f"{f['goals']['home']}-{f['goals']['away']}",
                        "winner": winner,
                    })
                except Exception: pass

    if fixture_id:
        inj_data = football_api("injuries", {"fixture": fixture_id}, ttl=600)
        if inj_data and inj_data.get("response"):
            for p in inj_data["response"]:
                try:
                    team_id = p.get("team", {}).get("id", 0)
                    entry = {"player": p["player"]["name"], "reason": p["player"].get("reason", "Baja")}
                    if team_id == h_id:
                        deep_data["injuries_home"].append(entry)
                    elif team_id == a_id:
                        deep_data["injuries_away"].append(entry)
                except Exception: pass

    return deep_data

# ═══════════════════════════════════════════════════════════
#  MÓDULO CUANTITATIVO (V11 §2.2) — PURO PYTHON, SIN LLM
#  Poisson bivariado con corrección Dixon-Coles para marcadores
#  bajos. El LLM recibe estos números YA CALCULADOS y su papel
#  pasa a ser interpretarlos, no inventarlos.
# ═══════════════════════════════════════════════════════════
LEAGUE_AVG_GOALS_HOME = 1.50   # priors razonables de fútbol profesional;
LEAGUE_AVG_GOALS_AWAY = 1.20   # se sustituyen por datos reales cuando hay stats
DIXON_COLES_RHO       = -0.10  # correlación empates bajos (valor típico de la literatura)

def get_team_season_stats(team_id: int, league_id: int) -> Optional[dict]:
    """Medias reales de goles marcados/recibidos (casa/fuera) de la temporada
    actual vía API-Football. Sustituye a las constantes mágicas: se recalcula
    solo con cada partido nuevo (cache 6h)."""
    if not team_id or not league_id:
        return None
    season = now_spain().year if now_spain().month >= 7 else now_spain().year - 1
    data = football_api("teams/statistics",
                        {"team": team_id, "league": league_id, "season": season}, ttl=21600)
    if not data or not data.get("response"):
        return None
    try:
        r = data["response"]
        gf, ga = r["goals"]["for"]["average"], r["goals"]["against"]["average"]
        played = r["fixtures"]["played"]
        if int(played.get("total") or 0) < 4:
            return None   # muestra insuficiente: mejor sin dato que dato ruidoso
        return {
            "gf_home": float(gf.get("home") or 0), "gf_away": float(gf.get("away") or 0),
            "ga_home": float(ga.get("home") or 0), "ga_away": float(ga.get("away") or 0),
            "played_total": int(played.get("total") or 0),
        }
    except Exception:
        return None

def _poisson_pmf(lmbda: float, k: int) -> float:
    return math.exp(-lmbda) * lmbda ** k / math.factorial(k)

def _dc_tau(x: int, y: int, lh: float, la: float, rho: float = DIXON_COLES_RHO) -> float:
    """Corrección Dixon-Coles: los marcadores 0-0/1-0/0-1/1-1 no son
    independientes como asume Poisson puro."""
    if x == 0 and y == 0: return 1 - lh * la * rho
    if x == 0 and y == 1: return 1 + lh * rho
    if x == 1 and y == 0: return 1 + la * rho
    if x == 1 and y == 1: return 1 - rho
    return 1.0

def quant_poisson_analysis(deep_data: dict, match: dict) -> dict:
    """Devuelve xG por equipo y probabilidades de mercado calculadas.
    Prioridad de fuentes: stats reales de temporada → media H2H → None."""
    out = {"disponible": False, "fuente": None}
    league_id = match.get("league_id", 0)
    h_id = deep_data.get("home_team_id", 0)
    a_id = deep_data.get("away_team_id", 0)

    xg_h = xg_a = None
    stats_h = get_team_season_stats(h_id, league_id)
    stats_a = get_team_season_stats(a_id, league_id)

    if stats_h and stats_a:
        # fuerza ataque × debilidad defensa relativa a la media de liga
        atk_h = stats_h["gf_home"] / max(LEAGUE_AVG_GOALS_HOME, 0.1)
        def_a = stats_a["ga_away"] / max(LEAGUE_AVG_GOALS_HOME, 0.1)
        atk_a = stats_a["gf_away"] / max(LEAGUE_AVG_GOALS_AWAY, 0.1)
        def_h = stats_h["ga_home"] / max(LEAGUE_AVG_GOALS_AWAY, 0.1)
        xg_h = max(0.2, min(4.5, LEAGUE_AVG_GOALS_HOME * atk_h * def_a))
        xg_a = max(0.1, min(4.0, LEAGUE_AVG_GOALS_AWAY * atk_a * def_h))
        out["fuente"] = f"stats temporada real ({stats_h['played_total']}+{stats_a['played_total']} partidos)"
    else:
        # fallback: media de goles del H2H real (últimos 5)
        h2h = deep_data.get("h2h", [])
        goles = []
        for f in h2h:
            try:
                gl, gv = f["score"].split("-")
                goles.append((int(gl), int(gv)))
            except Exception:
                pass
        if len(goles) >= 3:
            xg_h = max(0.2, statistics.mean(g[0] for g in goles))
            xg_a = max(0.1, statistics.mean(g[1] for g in goles))
            out["fuente"] = f"media H2H real ({len(goles)} partidos)"

    if xg_h is None or xg_a is None:
        return out   # sin datos suficientes: mejor declarar 'no disponible' que inventar

    # matriz de marcadores 0..8 con corrección Dixon-Coles
    p_home = p_draw = p_away = p_o25 = p_btts = 0.0
    best_score, best_p = "1-1", 0.0
    for x in range(9):
        for y in range(9):
            p = _poisson_pmf(xg_h, x) * _poisson_pmf(xg_a, y) * _dc_tau(x, y, xg_h, xg_a)
            p = max(p, 0.0)
            if x > y:  p_home += p
            elif x == y: p_draw += p
            else: p_away += p
            if x + y > 2: p_o25 += p
            if x > 0 and y > 0: p_btts += p
            if p > best_p:
                best_p, best_score = p, f"{x}-{y}"
    total = p_home + p_draw + p_away
    if total <= 0:
        return out
    p_home, p_draw, p_away = p_home/total, p_draw/total, p_away/total
    p_o25, p_btts = min(p_o25/total, 1.0), min(p_btts/total, 1.0)

    out.update({
        "disponible": True,
        "xg_local": round(xg_h, 2), "xg_visitante": round(xg_a, 2),
        "goles_esperados_total": round(xg_h + xg_a, 2),
        "p_local": round(p_home, 3), "p_empate": round(p_draw, 3), "p_visitante": round(p_away, 3),
        "p_over_2_5": round(p_o25, 3), "p_btts": round(p_btts, 3),
        "marcador_mas_probable": best_score,
    })
    return out

def format_quant_block(quant: dict) -> str:
    """Bloque de texto con los números calculados, para inyectar en prompts
    y en el mensaje final."""
    if not quant.get("disponible"):
        return "MODELO CUANTITATIVO: sin datos suficientes (no usar cifras inventadas)."
    return (
        f"MODELO CUANTITATIVO (Poisson/Dixon-Coles sobre datos reales — {quant['fuente']}):\n"
        f"  xG local={quant['xg_local']} | xG visitante={quant['xg_visitante']} | total={quant['goles_esperados_total']}\n"
        f"  P(1)={quant['p_local']:.0%}  P(X)={quant['p_empate']:.0%}  P(2)={quant['p_visitante']:.0%}\n"
        f"  P(Más de 2.5 goles)={quant['p_over_2_5']:.0%}  P(BTTS)={quant['p_btts']:.0%}\n"
        f"  Marcador más probable: {quant['marcador_mas_probable']}"
    )

# ═══════════════════════════════════════════════════════════
#  MULTI-SCRAPER — 4 FUENTES INDEPENDIENTES CON VALIDACIÓN
# ═══════════════════════════════════════════════════════════
_scrape_cache: dict = {}
_scrape_cache_ts: dict = {}
SCRAPE_TTL = 1800

SCRAPE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.6367.82 Mobile Safari/537.36",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Cache-Control": "no-cache",
}

SCRAPE_HEADERS_JSON = {
    **SCRAPE_HEADERS,
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.sofascore.com/",
    "Origin": "https://www.sofascore.com",
}

from curl_cffi import requests as curl_requests

def _get(url: str, timeout: int = 15, headers: dict = None, is_json: bool = False) -> Optional[str | dict]:
    try:
        h = headers or (SCRAPE_HEADERS_JSON if is_json else SCRAPE_HEADERS)
        r = curl_requests.get(url, headers=h, timeout=timeout, impersonate="chrome110")
        if r.status_code != 200:
            logger.debug(f"Bloqueo o error {r.status_code} en {url}")
            return None
        if is_json:
            return r.json()
        return r.text
    except Exception as e:
        logger.debug(f"GET error {url}: {e}")
        return None

def _normalize_name(name: str) -> str:
    name = name.lower().strip()
    for prefix in ["fc ", "cf ", "rcd ", "afc ", "rc ", "ss ", "as ", "ac ", "atletico ", "atlético "]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name

def _teams_match(a: str, b: str, threshold: int = 5) -> bool:
    na, nb = _normalize_name(a), _normalize_name(b)
    if na == nb: return True
    if na in nb or nb in na: return True
    if len(na) >= threshold and len(nb) >= threshold:
        if na[:threshold] == nb[:threshold]: return True
    return False

_SOFA_REASON = {
    1: "Lesión", 2: "Suspensión", 3: "Selección nacional",
    4: "No convocado", 5: "Duda", 6: "Personal",
}

def _sofascore_get_event(home: str, away: str) -> Optional[dict]:
    today = now_spain().strftime("%Y-%m-%d")
    data = _get(f"https://www.sofascore.com/api/v1/sport/football/scheduled-events/{today}", is_json=True)
    if not data or not isinstance(data, dict):
        return None

    for ev in data.get("events", []):
        hn = ev.get("homeTeam", {}).get("name", "")
        an = ev.get("awayTeam", {}).get("name", "")
        if _teams_match(home, hn) and _teams_match(away, an):
            ev_id = ev.get("id")
            if ev_id:
                detail = _get(f"https://www.sofascore.com/api/v1/event/{ev_id}", is_json=True)
                if detail:
                    ev_full = detail.get("event", detail)
                    if ev_full.get("referee") and ev_full["referee"].get("id"):
                        ev["referee"] = ev_full["referee"]
            return ev
    return None

def _sofascore_scrape(home: str, away: str) -> dict:
    result = {}
    try:
        ev = _sofascore_get_event(home, away)
        if not ev:
            return result

        ev_id   = ev.get("id")
        h_id_ev = ev.get("homeTeam", {}).get("id")
        a_id_ev = ev.get("awayTeam", {}).get("id")

        venue = ev.get("venue", {})
        if isinstance(venue, dict):
            v_name = venue.get("name") or venue.get("stadium", {}).get("name", "")
            if v_name:
                result["estadio"] = v_name.strip()

        ref = ev.get("referee", {})
        if isinstance(ref, dict) and ref.get("name"):
            result["arbitro"] = ref["name"].strip()
            games = ref.get("games", 0)
            if games and games > 0:
                yc  = ref.get("yellowCards", 0) or 0
                rc  = ref.get("redCards", 0) or 0
                yrc = ref.get("yellowRedCards", 0) or 0
                avg_cards = round((yc + rc + yrc) / games, 1)
                result["arbitro_tarjetas_partido"] = avg_cards
                result["arbitro_estilo"] = (
                    "estricto" if avg_cards > 5 else
                    ("permisivo" if avg_cards < 3 else "equilibrado")
                )
                result["arbitro_historial_fuente"] = "SofaScore"

        if ev_id:
            h2h_data = _get(f"https://www.sofascore.com/api/v1/event/{ev_id}/h2h", is_json=True)
            if h2h_data and isinstance(h2h_data, dict):
                duel = h2h_data.get("teamDuel", {})
                hw = duel.get("homeWins", 0) or 0
                aw = duel.get("awayWins", 0) or 0
                dr = duel.get("draws", 0) or 0
                total = hw + aw + dr
                if total > 0:
                    home_name = ev.get("homeTeam", {}).get("name", home)
                    away_name = ev.get("awayTeam", {}).get("name", away)
                    result["h2h_resumen"] = (
                        f"{home_name} {hw}V / {dr}E / {aw}V {away_name} "
                        f"(últimos {total} enfrentamientos)"
                    )
                    result["h2h_fuente"] = "SofaScore"

        if ev_id:
            lu = _get(f"https://www.sofascore.com/api/v1/event/{ev_id}/lineups", is_json=True)
            if lu and isinstance(lu, dict):

                for side, baja_key in [("home", "bajas_local"), ("away", "bajas_visitante")]:
                    missing = lu.get(side, {}).get("missingPlayers", [])
                    nombres = []
                    for mp in missing[:8]:
                        try:
                            pname = mp.get("player", {}).get("name", "")
                            desc = mp.get("description", "")
                            if not desc:
                                desc = _SOFA_REASON.get(mp.get("reason", 0), "Baja")
                            if pname:
                                nombres.append(f"{pname} ({desc})")
                        except Exception:
                            pass
                    if nombres:
                        result[baja_key] = ", ".join(nombres)
                        result[f"{baja_key}_fuente"] = "SofaScore"

                def _fmt(team_data):
                    formation = team_data.get("formation", "?")
                    players = [
                        p["player"]["name"]
                        for p in team_data.get("players", [])
                        if p.get("substitute") is False and p.get("player", {}).get("name")
                    ][:11]
                    return {"formation": formation, "players": players} if players else None

                h_lu = lu.get("home") or lu.get("homeTeam") or {}
                a_lu = lu.get("away") or lu.get("awayTeam") or {}
                h_fmt = _fmt(h_lu)
                a_fmt = _fmt(a_lu)
                if h_fmt:
                    result["alineacion_local"] = h_fmt
                if a_fmt:
                    result["alineacion_visitante"] = a_fmt

            h_id = ev.get("homeTeam", {}).get("id")
            a_id = ev.get("awayTeam", {}).get("id")

            for team_id, key in [(h_id, "forma_local"), (a_id, "forma_visitante")]:
                if not team_id:
                    continue
                recent = _get(
                    f"https://www.sofascore.com/api/v1/team/{team_id}/events/last/0",
                    is_json=True
                )
                if recent and isinstance(recent, dict):
                    events = recent.get("events", [])[:6]
                    forma = []
                    for fe in events:
                        wc = fe.get("winnerCode")
                        ht_id = fe.get("homeTeam", {}).get("id")
                        if wc == 1:
                            forma.append("V" if ht_id == team_id else "D")
                        elif wc == 2:
                            forma.append("D" if ht_id == team_id else "V")
                        elif wc == 3:
                            forma.append("E")
                    if forma:
                        result[key] = " ".join(forma[:5])

    except Exception as e:
        logger.debug(f"SofaScore error: {e}")

    return result

def _espn_scrape(home: str, away: str, league: str) -> dict:
    result = {}
    try:
        for league_code, league_name in ESPN_LEAGUES.items():
            if league and league.lower() not in league_name.lower() and league_name.lower() not in league.lower():
                if len(ESPN_LEAGUES) > 3:
                    continue

            url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league_code}/scoreboard"
            data = _get(url, is_json=True)
            if not data:
                continue

            for event in data.get("events", []):
                comps = event.get("competitions", [{}])[0]
                competitors = comps.get("competitors", [])
                h = next((c for c in competitors if c["homeAway"] == "home"), None)
                a = next((c for c in competitors if c["homeAway"] == "away"), None)
                if not h or not a:
                    continue

                h_name = h["team"].get("displayName", "")
                a_name = a["team"].get("displayName", "")

                if not (_teams_match(home, h_name) and _teams_match(away, a_name)):
                    continue

                officials = comps.get("officials", [])
                for official in officials:
                    if official.get("position", {}).get("name", "").lower() in ("referee", "árbitro", "arbitro"):
                        result["arbitro"] = official.get("displayName", "")
                        break

                venue = comps.get("venue", {})
                if venue.get("fullName"):
                    result["estadio"] = venue["fullName"]

                roster_url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league_code}/summary?event={event['id']}"
                roster_data = _get(roster_url, is_json=True)
                if roster_data:
                    rosters = roster_data.get("rosters", [])
                    for roster in rosters:
                        side = roster.get("homeAway", "")
                        athletes = roster.get("athletes", [])
                        starters = [
                            a_info.get("displayName", "")
                            for a_info in athletes
                            if a_info.get("starter") and a_info.get("displayName")
                        ]
                        if starters and side == "home":
                            result["alineacion_local"] = {"players": starters, "formation": roster.get("formation", "?")}
                        elif starters and side == "away":
                            result["alineacion_visitante"] = {"players": starters, "formation": roster.get("formation", "?")}

                return result
    except Exception as e:
        logger.debug(f"ESPN scrape error: {e}")

    return result

def _apifootball_form(fixture_id: int, league_id: int, home_id: int = None, away_id: int = None) -> dict:
    result = {}
    if API_FOOTBALL_EXHAUSTED or not FOOTBALL_API_KEY:
        return result
    try:
        stats = football_api("fixtures/statistics", {"fixture": fixture_id}, ttl=300)
        if stats and stats.get("response"):
            for team_stats in stats["response"]:
                team_name = team_stats.get("team", {}).get("name", "")
                stats_dict = {}
                for stat in team_stats.get("statistics", []):
                    stats_dict[stat["type"]] = stat["value"]
                result[f"stats_{team_name[:10]}"] = stats_dict

        if league_id and league_id != 0:
            season = now_spain().year
            stand = football_api("standings", {"league": league_id, "season": season}, ttl=3600)
            if stand and stand.get("response"):
                for group in stand["response"]:
                    for league_group in group.get("league", {}).get("standings", []):
                        for team in league_group:
                            t_name = team.get("team", {}).get("name", "")
                            form = team.get("form", "")
                            if form and len(form) >= 3:
                                result[f"form_{t_name[:12]}"] = form[-5:]
    except Exception as e:
        logger.debug(f"API-Football form error: {e}")
    return result

def scrape_match_context(home: str, away: str, league: str, fixture_id: int = 0, league_id: int = 0) -> dict:
    cache_key = f"{home}|{away}|{league}"
    now_ts = time.time()
    if cache_key in _scrape_cache and now_ts - _scrape_cache_ts.get(cache_key, 0) < SCRAPE_TTL:
        return _scrape_cache[cache_key]

    logger.info(f"Scraping multi-fuente: {home} vs {away}")

    sofa = _sofascore_scrape(home, away)
    espn = _espn_scrape(home, away, league)
    apifb = _apifootball_form(fixture_id, league_id) if fixture_id else {}

    arbitro = None
    arbitro_fuente = []
    if sofa.get("arbitro"):
        arbitro = sofa["arbitro"]
        arbitro_fuente.append("SofaScore")
    if espn.get("arbitro"):
        if arbitro and _normalize_name(arbitro).split()[0] in _normalize_name(espn["arbitro"]):
            arbitro_fuente.append("ESPN")
        elif not arbitro:
            arbitro = espn["arbitro"]
            arbitro_fuente.append("ESPN")

    estadio = None
    estadio_fuente = []
    if sofa.get("estadio"):
        estadio = sofa["estadio"]
        estadio_fuente.append("SofaScore")
    if espn.get("estadio"):
        if not estadio:
            estadio = espn["estadio"]
            estadio_fuente.append("ESPN")
        else:
            estadio_fuente.append("ESPN")

    alin_local = sofa.get("alineacion_local") or espn.get("alineacion_local")
    alin_visit = sofa.get("alineacion_visitante") or espn.get("alineacion_visitante")
    alin_fuente = []
    if sofa.get("alineacion_local"): alin_fuente.append("SofaScore")
    elif espn.get("alineacion_local"): alin_fuente.append("ESPN")

    forma_local = sofa.get("forma_local")
    forma_visit = sofa.get("forma_visitante")

    if not forma_local and apifb:
        for key, val in apifb.items():
            if key.startswith("form_") and home.lower()[:5] in key.lower():
                forma_local = val
            if key.startswith("form_") and away.lower()[:5] in key.lower():
                forma_visit = val

    result = {
        "arbitro": arbitro,
        "arbitro_confirmado": len(arbitro_fuente) >= 2,
        "arbitro_fuentes": arbitro_fuente,
        "estadio": estadio,
        "estadio_fuentes": estadio_fuente,
        "alineacion_local": alin_local,
        "alineacion_visitante": alin_visit,
        "alineacion_fuentes": alin_fuente,
        "alineaciones_confirmadas": bool(alin_local and alin_visit),
        "forma_local": forma_local,
        "forma_visitante": forma_visit,
        "forma_confirmada": bool(forma_local and forma_visit),
        "forma_confirmada_scraping": bool(forma_local and forma_visit),
        "bajas": apifb.get("injuries", None),
        "arbitro_tarjetas_partido": sofa.get("arbitro_tarjetas_partido"),
        "arbitro_penaltis_partido": sofa.get("arbitro_penaltis_partido"),
        "arbitro_estilo": sofa.get("arbitro_estilo", ""),
        "arbitro_historial_fuente": sofa.get("arbitro_historial_fuente", ""),
        "h2h_resumen": sofa.get("h2h_resumen") or None,
        "h2h_goles_promedio": sofa.get("h2h_goles_promedio"),
        "h2h_fuente": sofa.get("h2h_fuente", ""),
        "bajas_local": sofa.get("bajas_local") or None,
        "bajas_visitante": sofa.get("bajas_visitante") or None,
        "bajas_local_fuente": sofa.get("bajas_local_fuente", ""),
        "bajas_visitante_fuente": sofa.get("bajas_visitante_fuente", ""),
        "fuentes_consultadas": list(set(
            (["SofaScore"] if (arbitro and "SofaScore" in arbitro_fuente) or sofa.get("alineacion_local") or sofa.get("forma_local") else []) +
            (["ESPN"] if (arbitro and "ESPN" in arbitro_fuente) or espn.get("alineacion_local") else []) +
            (["API-Football"] if apifb and any(k.startswith("form_") for k in apifb) else [])
        )),
        "timestamp": now_ts,
        "datos_suficientes": bool(arbitro or alin_local or forma_local),
    }

    logger.info(
        f"Scrape completado '{home} vs {away}': "
        f"árbitro={'✓' if arbitro else '✗'} "
        f"alin={'✓' if alin_local else '✗'} "
        f"forma={'✓' if forma_local else '✗'} "
        f"fuentes={result['fuentes_consultadas']}"
    )

    _scrape_cache[cache_key] = result
    _scrape_cache_ts[cache_key] = now_ts
    return result

# ═══════════════════════════════════════════════════════════
#  ENRIQUECIMIENTO VIA GROQ — MULTI-RONDA V9.0
# ═══════════════════════════════════════════════════════════

def _parse_groq_json(resp: str) -> dict:
    if not resp:
        return {}
    try:
        clean = resp.strip().replace("```json", "").replace("```", "").strip()
        start = clean.find("{")
        end   = clean.rfind("}") + 1
        if start == -1 or end == 0:
            return {}
        return json.loads(clean[start:end])
    except Exception:
        return {}


async def _groq_ronda1_patron_partido(home: str, away: str, league: str,
                                       alin_local: str, alin_visit: str,
                                       bajas_local: str, bajas_visit: str,
                                       h2h_goles_media: Optional[float] = None,
                                       comp_ctx: Optional[dict] = None,
                                       quant: Optional[dict] = None,
                                       temperature: float = 0.2) -> dict:
    # V11 §2.2: el LLM recibe las cifras YA calculadas por el modelo cuantitativo
    quant_restriccion = ""
    if quant and quant.get("disponible"):
        quant_restriccion = (
            f"\n📊 DATOS CUANTITATIVOS CALCULADOS (OBLIGATORIO usarlos como base — NO los inventes):\n"
            f"{format_quant_block(quant)}\n"
            f"→ Tu 'goles_esperados_total' DEBE estar a ±0.5 de {quant['goles_esperados_total']}.\n"
            f"→ Tu 'btts_probable' DEBE ser coherente con P(BTTS)={quant['p_btts']:.0%} "
            f"(true si ≥55%, false si ≤45%, tu criterio en zona gris).\n"
            f"→ Tu papel es INTERPRETAR estos números junto al contexto (bajas, alineaciones), no recalcularlos."
        )
    h2h_restriccion = ""
    if h2h_goles_media is not None:
        if h2h_goles_media < 2.5:
            lado = "Menos de 2.5 goles"
            contrario = "Más de 2.5"
        elif h2h_goles_media < 3.5:
            lado = "entre 2.5 y 3.5 goles (zona media)"
            contrario = "Más de 3.5"
        else:
            lado = "Más de 3.5 goles"
            contrario = "Menos de 3.5"
        h2h_restriccion = (
            f"\n⚠️ RESTRICCIÓN NUMÉRICA OBLIGATORIA: La media H2H histórica de goles es {h2h_goles_media:.1f}."
            f"\n  → Tu estimación de goles_esperados_total DEBE ser coherente con {h2h_goles_media:.1f}."
            f"\n  → El mercado correcto de goles es '{lado}', NO '{contrario}'."
            f"\n  → No puedes recomendar un umbral al mismo lado que la media histórica."
        )

    comp_restriccion = ""
    if comp_ctx and comp_ctx.get("aviso_prompt"):
        tarj_adj = comp_ctx["tarjetas_ajuste"]
        fase_det = comp_ctx.get("fase_detectada", "")
        if tarj_adj <= -0.5:
            comp_restriccion = (
                f"\n⚠️ CONTEXTO COMPETICIONAL OBLIGATORIO: Este partido es {fase_det or league}. "
                f"En esta fase los jugadores EVITAN tarjetas (riesgo de suspensión). "
                f"Tu estimación de tarjetas_esperadas DEBE aplicar un ajuste de {tarj_adj:+.1f} "
                f"sobre la media habitual del árbitro. Este ajuste es NO NEGOCIABLE."
            )
        elif tarj_adj >= 0.5:
            comp_restriccion = (
                f"\n⚠️ CONTEXTO COMPETICIONAL: {fase_det or league} — "
                f"se esperan más tarjetas de lo habitual. Ajuste: {tarj_adj:+.1f}."
            )

    prompt = f"""Eres un analista de datos deportivos experto en patrones de partidos de fútbol. Tu objetivo es predecir con alta precisión los patrones estadísticos esperados basándote en datos históricos y contexto competitivo.

PARTIDO: {home} vs {away}
COMPETICIÓN: {league}
ALINEACIÓN LOCAL: {alin_local}
ALINEACIÓN VISITANTE: {alin_visit}
BAJAS LOCAL: {bajas_local}
BAJAS VISITANTE: {bajas_visit}{quant_restriccion}{h2h_restriccion}{comp_restriccion}

TAREA: Analiza los PATRONES ESPERADOS de este partido basándote en los datos cuantitativos calculados (si están disponibles, son tu base obligatoria), el estilo histórico de estos equipos, el contexto competitivo y cualquier restricción numérica obligatoria.

INSTRUCCIONES DE FORMATO:
- Responde EXCLUSIVAMENTE con un objeto JSON válido. No incluyas texto explicativo, markdown ni ningún otro contenido fuera del JSON.
- Si no tienes suficiente certeza sobre un campo, usa null para tipos numéricos/booleano o "" para strings.
- Los valores deben ser realistas y basados en datos históricos típicos del fútbol profesional.

EJEMPLO DE FORMATO ESPERADO:
{{
  "estilo_local": "Posesión paciente, busca juego por bandas",
  "estilo_visitante": "Contraataque rápido, transiciones verticales",
  "ritmo_esperado": "medio",
  "goles_esperados_total": 2.3,
  "corners_esperados_total": 10,
  "corners_local_mayor": false,
  "corners_confianza": "media",
  "senal_corners": "dudosa",
  "tarjetas_esperadas": 3.2,
  "partido_abierto": true,
  "primer_gol_antes_30": true,
  "btts_probable": true,
  "penalti_probable": false,
  "observaciones_clave": "El árbitro tiende a ser permisivo; Local fuerte en segundas partes",
  "confianza": "alta"
}}

RESPUESTA OBLIGATORIA (JSON puro sin markdown):
{{
  "estilo_local": "<descripción breve del estilo de juego típico del equipo local>",
  "estilo_visitante": "<descripción breve del estilo de juego típico del equipo visitante>",
  "ritmo_esperado": "<alto|medio|bajo> (alto: muchos cambios de posesión, bajo: juego posicional lento)",
  "goles_esperados_total": <número decimal entre 0.0 y 5.0 o null>,
  "corners_esperados_total": <número entero entre 0 y 20 o null>,
  "corners_local_mayor": <true si se espera que el local tenga más corners, false si el visitante, null si igual o desconocido>,
  "corners_confianza": "<alta|media|baja> — alta solo si conoces el promedio real de corners de ambos equipos esta temporada; media si estimas por estilo; baja si es pura suposición>",
  "senal_corners": "<clara|dudosa|sin_señal> — clara solo si corners_confianza=alta Y corners_esperados_total es claramente >11 o <8; dudosa si solo una condición; sin_señal en lo demás>",
  "tarjetas_esperadas": <número decimal entre 0.0 y 6.0 o null>,
  "partido_abierto": <true si se espera un juego con muchos espacios y ocasiones, false si cerrado y táctico, null si desconocido>,
  "primer_gol_antes_30": <true si hay probabilidad >50% de gol antes del minuto 30, false si <50%, null si desconocido>,
  "btts_probable": <true si es probable que ambos equipos marquen, false si no, null si desconocido>,
  "penalti_probable": <true si hay probabilidad >30% de penalti, false si <30%, null si desconocido>,
  "observaciones_clave": "<1-2 observaciones específicas y relevantes para el partido, máximo 150 caracteres>",
  "confianza": "<alta|media|baja> basada en la calidad de los datos disponibles y claridad de los patrones>
}}
CRÍTICO: Si no conoces con suficiente certeza un valor, usa null (o "" para strings). Nunca inventes datos.
{h2h_restriccion}{comp_restriccion}
⚠️ RECORDATORIO FINAL: Tu respuesta debe ser ÚNICAMENTE el objeto JSON. Nada más ni nada menos."""

    data = await ask_groq_json(prompt, schema=SCHEMA_RONDA1, temperature=temperature)
    logger.info(f"Groq R1 patrón '{home} vs {away}': confianza={data.get('confianza','?')}")
    return data


async def _groq_ronda1_self_consistency(home: str, away: str, league: str,
                                        alin_local: str, alin_visit: str,
                                        bajas_local: str, bajas_visit: str,
                                        h2h_goles_media: Optional[float],
                                        comp_ctx: Optional[dict],
                                        quant: Optional[dict],
                                        n: int = SELF_CONSISTENCY_N) -> dict:
    """V11 §2.3 — Self-consistency: ejecuta la Ronda 1 `n` veces con temperatura
    moderada y agrega: mediana en numéricos, moda en booleanos/strings.
    Alta dispersión entre pasadas ⇒ el modelo no tiene evidencia suficiente
    ⇒ la confianza se degrada AUTOMÁTICAMENTE (no se delega en la ronda 3)."""
    if n <= 1:
        return await _groq_ronda1_patron_partido(
            home, away, league, alin_local, alin_visit, bajas_local, bajas_visit,
            h2h_goles_media=h2h_goles_media, comp_ctx=comp_ctx, quant=quant)

    tareas = [
        _groq_ronda1_patron_partido(
            home, away, league, alin_local, alin_visit, bajas_local, bajas_visit,
            h2h_goles_media=h2h_goles_media, comp_ctx=comp_ctx, quant=quant,
            temperature=0.5)   # temperatura moderada: queremos variabilidad informativa
        for _ in range(n)
    ]
    resultados = [r for r in await asyncio.gather(*tareas, return_exceptions=True)
                  if isinstance(r, dict) and r]
    if not resultados:
        return {}
    if len(resultados) == 1:
        return resultados[0]

    NUMERICOS = ("goles_esperados_total", "corners_esperados_total", "tarjetas_esperadas")
    agregado: dict = {}
    dispersion_alta = False

    for campo in SCHEMA_RONDA1:
        valores = [r.get(campo) for r in resultados if r.get(campo) is not None]
        if not valores:
            agregado[campo] = None
            continue
        if campo in NUMERICOS:
            med = statistics.median(valores)
            agregado[campo] = round(float(med), 2)
            if len(valores) >= 2 and med > 0:
                cv = statistics.pstdev(valores) / med   # coeficiente de variación
                if cv > 0.25:
                    dispersion_alta = True
                    logger.info(f"Self-consistency: dispersión alta en {campo} "
                                f"(cv={cv:.2f}, valores={valores})")
        else:
            try:
                agregado[campo] = statistics.mode(valores)
            except statistics.StatisticsError:
                agregado[campo] = valores[0]

    agregado["selfconsistency_n"] = len(resultados)
    if dispersion_alta:
        agregado["confianza"] = "baja"
        agregado["selfconsistency_dispersion"] = "alta"
        metric_inc("selfconsistency_baja_confianza")
    return agregado


async def _groq_ronda2_scout_jugadores(home: str, away: str, league: str,
                                        alin_local: str, alin_visit: str,
                                        bajas_local: str, bajas_visit: str) -> dict:
    prompt = f"""Eres un scout profesional de fútbol europeo con acceso a estadísticas detalladas de la temporada actual. Tu objetivo es identificar jugadores con potencial alto para mercados específicos de apuestas basándote en su forma reciente, historial y role en el equipo.

PARTIDO: {home} vs {away} — {league}
ALINEACIÓN LOCAL ({home}): {alin_local}
ALINEACIÓN VISITANTE ({away}): {alin_visit}
BAJAS LOCAL: {bajas_local}
BAJAS VISITANTE: {bajas_visit}

TAREA CRÍTICA: Analiza CADA jugador de ambas alineaciones por su nombre real y su historial esta temporada. Identifica quienes tienen mayor probabilidad de destacar en mercados específicos.

INSTRUCCIONES DE FORMATO:
- Responde EXCLUSIVAMENTE con un objeto JSON válido. No incluyas texto explicativo, markdown ni ningún otro contenido fuera del JSON.
- Usa null o arrays vacíos si no encuentras jugadores que cumplan con los criterios.
- Los nombres deben ser los oficiales tal como aparecen en las alineaciones.
- Los motivos deben ser estadísticas concretas de la temporada actual/pasada reciente.
- Sé específico y basado en datos, no en especulaciones.

EJEMPLO DE FORMATO ESPERADO:
{{
  "goleadores_calientes": [
    {{"jugador": "Erling Haaland", "equipo": "local", "motivo": "1.2 goles por partido últimos 5", "mercado_sugerido": "Anytime Goalscorer", "valor": "alto"}},
    {{"jugador": "Mohamed Salah", "equipo": "visitante", "motivo": "0.8 goles por partido últimos 10", "mercado_sugerido": "Anytime Goalscorer", "valor": "medio"}}
  ],
  "asistentes_clave": [
    {{"jugador": "Kevin De Bruyne", "equipo": "local", "motivo": "0.6 asistencias por partido"}},
    {{"jugador": "Bruno Fernandes", "equipo": "visitante", "motivo": "Penalti taker principal"}}
  ],
  "portero_destacado": {{"jugador": "Alisson Becker", "equipo": "local", "motivo": "75% paradas últimos 10 partidos", "mercado": "Portero con menos de X goles"}},
  "lanzador_penaltis_local": "Lionel Messi",
  "lanzador_penaltis_visitante": "Karim Benzema",
  "jugadores_tarjeta_riesgo": [
    {{"jugador": "Sergio Ramos", "equipo": "local", "motivo": "8 amarillas en 12 partidos esta temporada", "amarillas_por_partido": 0.67, "posicion_riesgo": "central defensivo", "acumulacion_riesgo": true, "valor": "alto"}},
    {{"jugador": "Casemiro", "equipo": "visitante", "motivo": "Mediocentro defensivo con entradas duras, 4 amarillas en 8 partidos", "amarillas_por_partido": 0.5, "posicion_riesgo": "pivote", "acumulacion_riesgo": false, "valor": "medio"}}
  ],
  "senal_tarjetas_totales": "clara",
  "corners_ejecutor_local": "Trent Alexander-Arnold",
  "corners_ejecutor_visitante": "James Ward-Prowse",
  "tiros_a_puerta_lider_local": {{"jugador": "Vinicius Junior", "tiros_partido": 3.8}},
  "tiros_a_puerta_lider_visitante": {{"jugador": "Jude Bellingham", "tiros_partido": 3.2}},
  "jugador_impacto_ausencia": "Karim Benzema (ausencia reduce amenaza ofensiva un 40%)",
  "sorpresa_posible": "Rodrygo Goes (podría tener minutos de calidad desde banquillo)",
  "mercado_jugador_top": {{"mercado": "Haaland 2+ goles", "jugador": "Erling Haaland", "cuota_estimada": 3.5, "motivo": "En racha, enfrenta defensa débil"}},
  "confianza_jugadores": "alta"
}}

RESPUESTA OBLIGATORIA (JSON puro sin markdown):
{{
  "goleadores_calientes": [
    {{"jugador": "<nombre real>", "equipo": "<local|visitante>", "motivo": "<estadística concreta de temporada>", "mercado_sugerido": "<tipo de mercado de apuestas>", "valor": "<alto|medio|bajo>"}}
  ],
  "asistentes_clave": [
    {{"jugador": "<nombre real>", "equipo": "<local|visitante>", "motivo": "<estadística concreta>"}}
  ],
  "portero_destacado": {{"jugador": "<nombre real>", "equipo": "<local|visitante>", "motivo": "<forma reciente o habilidad destacada>", "mercado": "<mercado relacionado>"}},
  "lanzador_penaltis_local": "<nombre real del lanzador habitual o null si desconocido>",
  "lanzador_penaltis_visitante": "<nombre real del lanzador habitual o null si desconocido>",
  "jugadores_tarjeta_riesgo": [
    {{
      "jugador": "<nombre real>",
      "equipo": "<local|visitante>",
      "motivo": "<estadística disciplinaria concreta: X amarillas en Y partidos esta temporada>",
      "amarillas_por_partido": <decimal, ej 0.45, o null si no sabes>,
      "posicion_riesgo": "<central|lateral|pivote|mediapunta|delantero — posición con mayor riesgo de entrada>",
      "acumulacion_riesgo": <true si está cerca de sanción por acumulación, false si no, null si desconocido>,
      "valor": "<alto si amarillas_por_partido > 0.4 Y árbitro estricto | medio si amarillas_por_partido 0.2-0.4 | bajo si < 0.2>"
    }}
  ],
  "senal_tarjetas_totales": "<clara|dudosa|sin_señal> — clara solo si árbitro promedia >=4.5 tarjetas/partido Y partido de alta tensión; dudosa si solo una condición; sin_señal en lo demás>",
  "corners_ejecutor_local": "<nombre real del principal ejecutor de corners o null>",
  "corners_ejecutor_visitante": "<nombre real del principal ejecutor de corners o null>",
  "tiros_a_puerta_lider_local": {{"jugador": "<nombre real del que más tiros a puerta realiza>", "tiros_partido": <número decimal promedio por partido o null>}},
  "tiros_a_puerta_lider_visitante": {{"jugador": "<nombre real del que más tiros a puerta realiza>", "tiros_partido": <número decimal promedio por partido o null>}},
  "jugador_impacto_ausencia": "<nombre del jugador cuya ausencia afecta más al equipo + breve explicación del impacto o null>",
  "sorpresa_posible": "<nombre de jugador que podría tener un rendimiento inesperadamente bueno o null>",
  "mercado_jugador_top": {{"mercado": "<descripción específica del mercado de apuestas>", "jugador": "<nombre real>", "cuota_estimada": <número decimal representando la cuota>, "motivo": "<razón específica basada en datos>"}},
  "confianza_jugadores": "<alta|media|baja> basada en la disponibilidad y calidad de datos estadísticos>"
}}
IMPORTANTE: Intenta rellenar goleadores_calientes con AL MENOS 3 jugadores si los hay y los datos lo permiten. Si no hay suficientes datos, rellena con los que tengas o deja el array vacío.
⚠️ RECORDATORIO FINAL: Tu respuesta debe ser ÚNICAMENTE el objeto JSON. Nada más ni nada menos. Inventa NADA. Si no sabes, usa null o arrays vacíos."""

    data = await ask_groq_json(prompt, temperature=0.25)
    logger.info(f"Groq R2 scout '{home} vs {away}': goleadores={len(data.get('goleadores_calientes', []))}, tarjetas={len(data.get('jugadores_tarjeta_riesgo', []))}")
    return data


async def _groq_ronda3_datos_base(home: str, away: str, league: str,
                                   arbitro: str,
                                   patron: dict, scout: dict) -> dict:
    resumen_patron = (
        f"Goles esperados: {patron.get('goles_esperados_total', '?')} | "
        f"Corners: {patron.get('corners_esperados_total', '?')} | "
        f"BTTS probable: {patron.get('btts_probable', '?')} | "
        f"Partido abierto: {patron.get('partido_abierto', '?')}"
    )
    goleadores = ", ".join(
        f"{g['jugador']} ({g['equipo']}, {g['motivo']})"
        for g in scout.get("goleadores_calientes", [])[:3]
    ) or "ninguno identificado"

    prompt = f"""Eres una base de datos especializada en fútbol UEFA/FIFA con conocimiento detallado de árbitros, forma de equipos y historial head-to-head. Tu objetivo es proporcionar información factual y verificable para completar el análisis predictivo.

PARTIDO: {home} vs {away}
COMPETICIÓN: {league}
FECHA: {now_spain().strftime('%d/%m/%Y')}
ÁRBITRO: {arbitro or "desconocido"}
ANÁLISIS PREVIO PATRÓN: {resumen_patron}
GOLEADORES IDENTIFICADOS: {goleadores}

INSTRUCCIONES DE FORMATO:
- Responde EXCLUSIVAMENTE con un objeto JSON válido. No incluyas texto explicativo, markdown ni ningún otro contenido fuera del JSON.
- Si no tienes información verificable con certeza, usa null para tipos numéricos/booleano o "" para strings.
- Prioriza la precisión sobre la completitud: mejor null que información incorrecta.
- Los formatos deben seguir exactamente lo especificado.

EJEMPLO DE FORMATO ESPERADO:
{{
  "arbitro": "Anthony Taylor",
  "arbitro_tarjetas_partido": 4.2,
  "arbitro_estilo": "equilibrado",
  "arbitro_penaltis_partido": 0.3,
  "arbitro_corners_influencia": "no",
  "forma_local": "V V E D V",
  "forma_visitante": "D E V V E",
  "h2h_resumen": "Local 2-1, Empate 1-1, Visitante 0-3",
  "h2h_goles_promedio": 2.0,
  "h2h_corners_promedio": 9.5,
  "bajas_local": "Lesionado: Kevin De Bruyne (rodilla)",
  "bajas_visitante": "Suspendido: Casemiro (2 amarillas)",
  "prediccion_resultado_mas_probable": "1",
  "prediccion_marcador_exacto": "2-1",
  "senal_valor": "Local gana y ambos marcan",
  "confianza_datos": "media"
}}

RESPUESTA OBLIGATORIA (JSON puro sin markdown):
{{
  "arbitro": "<nombre completo del árbitro o null si no está confirmado>",
  "arbitro_tarjetas_partido": <número decimal promedio de tarjetas por partido en sus últimos 15 partidos o null>,
  "arbitro_estilo": "<estricto|permisivo|equilibrado o null> basado en tendencia a tarjetas",
  "arbitro_penaltis_partido": <número decimal promedio de penaltis señalados por partido o null>,
  "arbitro_corners_influencia": "<si|no|null> indicando si tiende a influir en número de corners",
  "forma_local": "<string con 5 caracteres: V=Victoria, E=Empate, D=Derrota, ej: VVE-DD o vacío si no hay datos>",
  "forma_visitante": "<string con 5 caracteres: V=Victoria, E=Empate, D=Derrota, o vacío si no hay datos>",
  "h2h_resumen": "<resumen de los últimos 3-5 enfrentamientos directos, formato: 'Local X-Y Visitante' o vacío>",
  "h2h_goles_promedio": <número decimal promedio de goles por partido en H2H recientes o null>,
  "h2h_corners_promedio": <número decimal promedio de corners por partido en H2H recientes o null>,
  "bajas_local": "<lista de jugadores ausentes con razón (lesión, suspensión, etc.) o vacío>",
  "bajas_visitante": "<lista de jugadores ausentes con razón (lesión, suspensión, etc.) o vacío>",
  "prediccion_resultado_mas_probable": "<1 para victoria local, X para empate, 2 para visita o null si demasiado incierto>",
  "prediccion_marcador_exacto": "<formato X-Y donde X=goles local, Y=goles visitante o null si muy incierto>",
  "senal_valor": "<descripción específica de mercado de apuestas con buena relación riesgo/recompensa basada en análisis o null>",
  "confianza_datos": "<alta|media|baja> basada en la verifiabilidad y fuentes de la información proporcionada>"
}}
CRÍTICO: Si no conoces con suficiente certeza un valor, usa null (o "" para strings). Nunca inventes datos ni especules sin base.
⚠️ RECORDATORIO FINAL: Tu respuesta debe ser ÚNICAMENTE el objeto JSON. Nada más ni nada menos. Sé conservador: cuando tengas dudas, usa null."""

    data = await ask_groq_json(prompt, temperature=0.2)
    logger.info(
        f"Groq R3 base '{home} vs {away}': "
        f"árbitro={'✓' if data.get('arbitro') else '✗'} "
        f"señal={data.get('senal_valor','ninguna')}"
    )
    return data


def _cove_verify_patron(patron: dict, quant: Optional[dict],
                        h2h_goles_media: Optional[float],
                        arb_tarjetas: Optional[float]) -> tuple[dict, list[str]]:
    """V11 §2.3 — Chain-of-Verification DETERMINISTA (por código, no por LLM):
    cada afirmación numérica del patrón se contrasta con los datos duros.
    Afirmación no soportada → se corrige al dato real o se anula, y se
    devuelve la lista de correcciones para mostrarla en el mensaje."""
    if not patron:
        return patron, []
    p = dict(patron)
    correcciones: list[str] = []

    # goles vs modelo cuantitativo (fuente más fiable disponible)
    ref_goles = quant.get("goles_esperados_total") if (quant and quant.get("disponible")) else h2h_goles_media
    g = p.get("goles_esperados_total")
    if ref_goles is not None and isinstance(g, (int, float)) and abs(g - ref_goles) > 0.8:
        correcciones.append(f"goles_esperados {g}→{ref_goles} (dato calculado, no soportaba {g})")
        p["goles_esperados_total"] = round(float(ref_goles), 2)

    # BTTS vs probabilidad calculada
    if quant and quant.get("disponible") and isinstance(p.get("btts_probable"), bool):
        pb = quant["p_btts"]
        if p["btts_probable"] and pb < 0.40:
            correcciones.append(f"btts_probable=true no soportado (P(BTTS)={pb:.0%}) → false")
            p["btts_probable"] = False
        elif not p["btts_probable"] and pb > 0.60:
            correcciones.append(f"btts_probable=false no soportado (P(BTTS)={pb:.0%}) → true")
            p["btts_probable"] = True

    # tarjetas vs media real del árbitro
    t = p.get("tarjetas_esperadas")
    if arb_tarjetas is not None and isinstance(t, (int, float)) and abs(t - arb_tarjetas) > 2.0:
        correcciones.append(f"tarjetas_esperadas {t}→{arb_tarjetas} (media real del árbitro)")
        p["tarjetas_esperadas"] = round(float(arb_tarjetas), 1)

    if correcciones:
        metric_inc("cove_correcciones", len(correcciones))
        p["confianza"] = "baja" if len(correcciones) >= 2 else p.get("confianza")
        logger.info(f"CoVe determinista: {len(correcciones)} correcciones — {correcciones}")
    return p, correcciones


async def enrich_with_groq(home: str, away: str, league: str, ctx: dict, deep_data: dict = None,
                           comp_ctx: Optional[dict] = None, quant: Optional[dict] = None) -> dict:
    arbitro_conocido   = ctx.get("arbitro", "")
    alin_local_str     = _format_lineup(ctx.get("alineacion_local")) if ctx.get("alineacion_local") else "NO DISPONIBLE"
    alin_visit_str     = _format_lineup(ctx.get("alineacion_visitante")) if ctx.get("alineacion_visitante") else "NO DISPONIBLE"
    if deep_data:
        api_lin = deep_data.get("lineups")
        if api_lin:
            alin_local_str  = f"({api_lin.get('home_formation','?')}): {', '.join(api_lin.get('home_players',[]))}"
            alin_visit_str  = f"({api_lin.get('away_formation','?')}): {', '.join(api_lin.get('away_players',[]))}"
    bajas_local_str    = ctx.get("bajas_local", "") or ""
    bajas_visit_str    = ctx.get("bajas_visitante", "") or ""
    h2h_goles_previa: Optional[float] = ctx.get("h2h_goles_promedio")

    patron = await _groq_ronda1_self_consistency(
        home, away, league, alin_local_str, alin_visit_str, bajas_local_str, bajas_visit_str,
        h2h_goles_media=h2h_goles_previa,
        comp_ctx=comp_ctx,
        quant=quant,
    )

    # CoVe determinista: correcciones de afirmaciones no soportadas por datos
    patron, cove_correcciones = _cove_verify_patron(
        patron, quant, h2h_goles_previa, ctx.get("arbitro_tarjetas_partido"))

    scout = await _groq_ronda2_scout_jugadores(
        home, away, league, alin_local_str, alin_visit_str, bajas_local_str, bajas_visit_str
    )

    groq_data = await _groq_ronda3_datos_base(
        home, away, league, arbitro_conocido, patron, scout
    )

    enriched = dict(ctx)

    if not enriched.get("arbitro") and groq_data.get("arbitro"):
        enriched["arbitro"] = groq_data["arbitro"]
        enriched["arbitro_fuentes"] = ["Groq/conocimiento"]
        enriched["arbitro_confirmado"] = False

    if enriched.get("arbitro_tarjetas_partido") is None:
        if groq_data.get("arbitro_tarjetas_partido") is not None:
            enriched["arbitro_tarjetas_partido"] = groq_data["arbitro_tarjetas_partido"]
            enriched["arbitro_historial_fuente"] = "Groq/conocimiento"
        if groq_data.get("arbitro_estilo"):
            enriched["arbitro_estilo"] = groq_data["arbitro_estilo"]
        if groq_data.get("arbitro_penaltis_partido") is not None:
            enriched["arbitro_penaltis_partido"] = groq_data["arbitro_penaltis_partido"]
    enriched["arbitro_corners_influencia"] = groq_data.get("arbitro_corners_influencia")

    if not enriched.get("forma_local") and groq_data.get("forma_local"):
        enriched["forma_local"] = groq_data["forma_local"]
    if not enriched.get("forma_visitante") and groq_data.get("forma_visitante"):
        enriched["forma_visitante"] = groq_data["forma_visitante"]
    if enriched.get("forma_local") and enriched.get("forma_visitante"):
        enriched["forma_confirmada"] = True

    if groq_data.get("h2h_resumen") and not enriched.get("h2h_resumen"):
        enriched["h2h_resumen"] = groq_data["h2h_resumen"]
        enriched["h2h_fuente"] = "Groq/conocimiento"
    if groq_data.get("h2h_goles_promedio") is not None and enriched.get("h2h_goles_promedio") is None:
        enriched["h2h_goles_promedio"] = groq_data["h2h_goles_promedio"]

    if groq_data.get("bajas_local") and not enriched.get("bajas_local"):
        enriched["bajas_local"] = groq_data["bajas_local"]
        enriched["bajas_local_fuente"] = "Groq/conocimiento"
    if groq_data.get("bajas_visitante") and not enriched.get("bajas_visitante"):
        enriched["bajas_visitante"] = groq_data["bajas_visitante"]
        enriched["bajas_visitante_fuente"] = "Groq/conocimiento"

    enriched["groq_patron"]        = patron
    enriched["groq_scout"]         = scout
    enriched["quant"]              = quant or {}
    enriched["cove_correcciones"]  = cove_correcciones
    enriched["groq_prediccion"] = {
        "resultado": groq_data.get("prediccion_resultado_mas_probable"),
        "marcador":  groq_data.get("prediccion_marcador_exacto"),
        "senal_valor": groq_data.get("senal_valor"),
        "h2h_corners_promedio": groq_data.get("h2h_corners_promedio"),
    }

    fuentes = enriched.get("fuentes_consultadas", [])
    if "Groq/conocimiento" not in fuentes:
        fuentes.append("Groq/conocimiento")
    enriched["fuentes_consultadas"] = fuentes
    enriched["groq_confianza"] = groq_data.get("confianza_datos", "")

    logger.info(
        f"Enriquecimiento multi-ronda '{home} vs {away}': "
        f"patrón_goles={patron.get('goles_esperados_total','?')} "
        f"goleadores={len(scout.get('goleadores_calientes',[]))} "
        f"señal='{groq_data.get('senal_valor','ninguna')}'"
    )
    return enriched

# ═══════════════════════════════════════════════════════════
#  FORMATEO DE DATOS
# ═══════════════════════════════════════════════════════════
def _format_lineup(lineup_data: Optional[dict]) -> str:
    if not lineup_data:
        return "NO DISPONIBLE"
    formation = lineup_data.get("formation", "?")
    players = lineup_data.get("players", [])
    if not players:
        return "NO DISPONIBLE"
    return f"({formation}): {', '.join(players[:11])}"

def _format_forma(forma: Optional[str]) -> str:
    if not forma:
        return "NO DISPONIBLE"
    valid = all(c in "VED- " for c in forma.upper())
    if not valid or len(forma.strip()) < 3:
        return "NO DISPONIBLE"
    return forma.upper()

def _confianza_dato(valor, fuentes: list, confirmado: bool) -> str:
    if not valor:
        return "❌ Sin datos"
    if confirmado or len(fuentes) >= 2:
        return f"✅ Confirmado ({', '.join(fuentes)})"
    return f"⚠️ 1 fuente ({', '.join(fuentes)})"

# ═══════════════════════════════════════════════════════════
#  PROMPTS TRADER
# ═══════════════════════════════════════════════════════════
async def analyze_phase_1(match: dict) -> str:
    league_id = match.get("league_id", 0)
    espn_code = match.get("espn_league_code", "")
    liga_nombre = str(match.get("league", "")).strip()

    if league_id and league_id in ALL_LEAGUES:
        logger.debug(f"Fase1 ESTELAR (API-Football ID={league_id}): {liga_nombre}")
        return "ESTELAR"

    if espn_code and espn_code in ESPN_LEAGUES:
        logger.debug(f"Fase1 ESTELAR (ESPN code={espn_code}): {liga_nombre}")
        return "ESTELAR"

    logger.debug(f"Fase1 RANDOM (sin origen verificado): {liga_nombre} — {match.get('home')} vs {match.get('away')}")
    return "RANDOM"


async def analyze_phase_2_trader(match: dict, deep_data: dict, is_manual=False) -> str:
    estado = f"EN VIVO (Min {match.get('minute','?')})" if match.get("is_live") else (
        "Faltan ~45 min" if not is_manual else "Pre-partido / Manual"
    )

    # ── V10: Contexto competicional ───────────────────────────────────────────
    round_info = match.get("round", "") or match.get("league_round", "") or ""
    comp_ctx = get_competition_context(match.get("league", ""), round_info)

    # Scraping en executor
    loop = asyncio.get_running_loop()
    ctx = await loop.run_in_executor(
        None, lambda: scrape_match_context(
            match["home"], match["away"], match["league"],
            fixture_id=match.get("fixture_id", 0),
            league_id=match.get("league_id", 0)
        )
    )

    # ── V11 §2.2: modelo cuantitativo PRIMERO (sin LLM, datos reales) ─────────
    quant = await loop.run_in_executor(None, lambda: quant_poisson_analysis(deep_data, match))
    if quant.get("disponible"):
        logger.info(f"Quant '{match['home']} vs {match['away']}': "
                    f"xG {quant['xg_local']}-{quant['xg_visitante']} "
                    f"P(1/X/2)={quant['p_local']:.0%}/{quant['p_empate']:.0%}/{quant['p_visitante']:.0%} "
                    f"[{quant['fuente']}]")

    # Enriquecimiento con Groq (el LLM interpreta las cifras del quant, no las inventa)
    ctx = await enrich_with_groq(match["home"], match["away"], match["league"], ctx, deep_data,
                                 comp_ctx=comp_ctx, quant=quant)

    # Árbitro
    arbitro_api = match.get("referee")
    arbitro_web = ctx.get("arbitro")
    if arbitro_api and arbitro_api not in ("Desconocido", "ESPN", "", None):
        arbitro_final = arbitro_api
        arbitro_status = "✅ API-Football"
    elif arbitro_web:
        confirmado = ctx.get("arbitro_confirmado", False)
        fuentes = ctx.get("arbitro_fuentes", [])
        arbitro_final = arbitro_web
        arbitro_status = _confianza_dato(arbitro_web, fuentes, confirmado)
    else:
        arbitro_final = None
        arbitro_status = "❌ Sin datos"

    # Estadio
    estadio_api = match.get("venue")
    estadio_web = ctx.get("estadio")
    if estadio_api and estadio_api not in ("Desconocido", "", None):
        estadio_final = estadio_api
        estadio_status = "✅ API-Football"
    elif estadio_web:
        estadio_final = estadio_web
        estadio_status = f"⚠️ Web ({', '.join(ctx.get('estadio_fuentes', []))})"
    else:
        estadio_final = None
        estadio_status = "❌ Sin datos"

    # Alineaciones
    api_lineups = deep_data.get("lineups")
    if api_lineups and isinstance(api_lineups, dict):
        alin_local_str = f"({api_lineups.get('home_formation','?')}): {', '.join(api_lineups.get('home_players', []))}"
        alin_visit_str = f"({api_lineups.get('away_formation','?')}): {', '.join(api_lineups.get('away_players', []))}"
        alin_status = "✅ API-Football"
    elif ctx.get("alineaciones_confirmadas"):
        alin_local_str = _format_lineup(ctx.get("alineacion_local"))
        alin_visit_str = _format_lineup(ctx.get("alineacion_visitante"))
        alin_status = f"⚠️ Web ({', '.join(ctx.get('alineacion_fuentes', []))})"
    else:
        alin_local_str = "NO DISPONIBLE (confirmar en ~60 min antes del partido)"
        alin_visit_str = "NO DISPONIBLE"
        alin_status = "❌ Sin datos"

    # Forma reciente
    forma_local = _format_forma(ctx.get("forma_local"))
    forma_visit = _format_forma(ctx.get("forma_visitante"))
    forma_status = "✅ Confirmada" if ctx.get("forma_confirmada") else "⚠️ Parcial" if (ctx.get("forma_local") or ctx.get("forma_visitante")) else "❌ Sin datos"

    # Nivel de datos
    groq_confianza = ctx.get("groq_confianza", "")
    groq_usada = "Groq/conocimiento" in ctx.get("fuentes_consultadas", [])
    groq_es_baja = groq_usada and groq_confianza in ("baja", "low", "")

    def _peso_forma(val, desde_groq_baja):
        if val == "NO DISPONIBLE": return 0
        return 0.5 if desde_groq_baja else 1

    forma_desde_groq = groq_usada and not ctx.get("forma_confirmada_scraping", False)

    datos_ok_raw = (
        (1 if arbitro_final else 0) +
        (1 if estadio_final else 0) +
        (1 if "NO DISPONIBLE" not in alin_local_str else 0) +
        _peso_forma(forma_local, forma_desde_groq and groq_es_baja) +
        _peso_forma(forma_visit, forma_desde_groq and groq_es_baja)
    )
    datos_ok = int(datos_ok_raw)
    nivel_datos = "ALTO" if datos_ok_raw >= 4 else "MEDIO" if datos_ok_raw >= 2 else "BAJO"

    if groq_es_baja and not arbitro_final and "NO DISPONIBLE" in alin_local_str:
        nivel_datos = "BAJO"
        datos_ok = min(datos_ok, 1)

    # Bajas
    api_injuries_home = deep_data.get("injuries_home", [])
    api_injuries_away = deep_data.get("injuries_away", [])
    if api_injuries_home:
        bajas_local_str = ", ".join(f"{p['player']} ({p['reason']})" for p in api_injuries_home[:6])
        bajas_local_src = "✅ API-Football"
    elif ctx.get("bajas_local"):
        bajas_local_str = ctx["bajas_local"]
        bajas_local_fuente_ctx = ctx.get("bajas_local_fuente", "")
        bajas_local_src = f"⚠️ {bajas_local_fuente_ctx}" if bajas_local_fuente_ctx else "⚠️ Groq/conocimiento"
    else:
        bajas_local_str = "NO DISPONIBLE"
        bajas_local_src = "❌ Sin datos"

    if api_injuries_away:
        bajas_visit_str = ", ".join(f"{p['player']} ({p['reason']})" for p in api_injuries_away[:6])
        bajas_visit_src = "✅ API-Football"
    elif ctx.get("bajas_visitante"):
        bajas_visit_str = ctx["bajas_visitante"]
        bajas_visit_fuente_ctx = ctx.get("bajas_visitante_fuente", "")
        bajas_visit_src = f"⚠️ {bajas_visit_fuente_ctx}" if bajas_visit_fuente_ctx else "⚠️ Groq/conocimiento"
    else:
        bajas_visit_str = "NO DISPONIBLE"
        bajas_visit_src = "❌ Sin datos"

    # H2H
    api_h2h = deep_data.get("h2h", [])
    if api_h2h:
        h2h_str = " | ".join(f"{r['home']} {r['score']} {r['away']} ({r['date'][:7]})" for r in api_h2h[:4])
        h2h_src = "✅ API-Football"
        h2h_goles_api = sum(int(r["score"].split("-")[0]) + int(r["score"].split("-")[1]) for r in api_h2h) / len(api_h2h) if api_h2h else None
    elif ctx.get("h2h_resumen"):
        h2h_str = ctx["h2h_resumen"]
        h2h_fuente_ctx = ctx.get("h2h_fuente", "")
        h2h_src = f"⚠️ {h2h_fuente_ctx}" if h2h_fuente_ctx else "⚠️ Groq/conocimiento"
        h2h_goles_api = ctx.get("h2h_goles_promedio")
    else:
        h2h_str = "NO DISPONIBLE"
        h2h_src = "❌ Sin datos"
        h2h_goles_api = None

    h2h_goles_str = f" (media goles H2H: {h2h_goles_api:.1f})" if h2h_goles_api else ""

    # Historial árbitro
    arb_tarjetas = ctx.get("arbitro_tarjetas_partido")
    arb_estilo   = ctx.get("arbitro_estilo", "")
    arb_penaltis = ctx.get("arbitro_penaltis_partido")
    if arb_tarjetas is not None:
        arb_hist_str = f"{arb_tarjetas:.1f} tarjetas/partido, estilo {arb_estilo or 'desconocido'}"
        if arb_penaltis is not None:
            arb_hist_str += f", {arb_penaltis:.2f} penaltis/partido"
        arb_hist_fuente = ctx.get("arbitro_historial_fuente", "")
        arb_hist_src = f"⚠️ {arb_hist_fuente}" if arb_hist_fuente else "⚠️ Groq/conocimiento"
    else:
        arb_hist_str = "NO DISPONIBLE"
        arb_hist_src = "❌ Sin datos"

    # ── V10: CALCULAR arb_tarjetas_ajustado AQUÍ, antes de usarlo en prompt_mega ──
    arb_tarjetas_ajustado = (
        (arb_tarjetas + comp_ctx["tarjetas_ajuste"])
        if arb_tarjetas is not None else None
    )

    # Etiquetas de origen
    forma_origen = "Groq/conocimiento (orientativo)" if forma_desde_groq else "SofaScore/ESPN (verificado)"
    aviso_groq_forma = f" ⚠️ orientativo (Groq/{groq_confianza})" if forma_desde_groq else ""
    aviso_groq_arb   = " ⚠️ orientativo (Groq)" if (arbitro_final and ctx.get("arbitro_fuentes") == ["Groq/conocimiento"]) else ""

    # Cabecera fija
    cabecera = (
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔥 TRADER ALERT: {match['home']} vs {match['away']}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        + (
            f"{'🚨' if comp_ctx['nivel_alerta'] == 'CRITICO' else '⚠️' if comp_ctx['nivel_alerta'] == 'ALTO' else '📋'} "
            f"COMPETICIÓN: {match['league']}"
            + (f" | FASE: {comp_ctx['fase_detectada']}" if comp_ctx.get('fase_detectada') else "")
            + (f" | Tarjetas esperadas: {comp_ctx['tarjetas_ajuste']:+.1f} vs media árbitro" if comp_ctx['tarjetas_ajuste'] != 0 else "")
            + "\n"
            if comp_ctx.get("aviso_prompt") else ""
        )
        + f"📋 DATOS VERIFICADOS:\n"
        f"  🏟 Estadio: {estadio_final or 'NO DISPONIBLE'} [{estadio_status}]\n"
        f"  👨‍⚖️ Árbitro: {arbitro_final or 'NO DISPONIBLE'} [{arbitro_status}]{aviso_groq_arb}\n"
        f"  📊 Árbitro historial [{arb_hist_src}]: {arb_hist_str}\n"
        f"  📋 Alineación local [{alin_status}]: {alin_local_str}\n"
        f"  📋 Alineación visit [{alin_status}]: {alin_visit_str}\n"
        f"  📈 Forma {match['home']}{aviso_groq_forma}: {forma_local}\n"
        f"  📈 Forma {match['away']}{aviso_groq_forma}: {forma_visit}\n"
        f"  🔄 H2H [{h2h_src}]: {h2h_str}{h2h_goles_str}\n"
        f"  🏥 Bajas {match['home']} [{bajas_local_src}]: {bajas_local_str}\n"
        f"  🏥 Bajas {match['away']} [{bajas_visit_src}]: {bajas_visit_str}\n"
        f"📊 CALIDAD DE DATOS: {nivel_datos} ({datos_ok}/5 campos en tiempo real)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    # Datos de las 3 rondas Groq
    patron  = ctx.get("groq_patron", {})
    scout   = ctx.get("groq_scout", {})
    pred    = ctx.get("groq_prediccion", {})

    patron_str = ""
    if patron:
        patron_str = (
            f"\nPATRÓN ESPERADO (Groq R1):"
            f"\n  Estilo local: {patron.get('estilo_local','?')} | Estilo visit: {patron.get('estilo_visitante','?')}"
            f"\n  Goles esperados: {patron.get('goles_esperados_total','?')} | Corners esp: {patron.get('corners_esperados_total','?')} (confianza: {patron.get('corners_confianza','?')}, señal: {patron.get('senal_corners','sin_señal')})"
            f"\n  BTTS probable: {patron.get('btts_probable','?')} | Partido abierto: {patron.get('partido_abierto','?')}"
            f"\n  Tarjetas esperadas: {patron.get('tarjetas_esperadas','?')} | Primer gol <30min: {patron.get('primer_gol_antes_30','?')}"
            f"\n  Penalti probable: {patron.get('penalti_probable','?')}"
            f"\n  Observación clave: {patron.get('observaciones_clave','')}"
        )

    scout_str = ""
    if scout:
        goleadores = scout.get("goleadores_calientes", [])
        tarjeta_riesgo = scout.get("jugadores_tarjeta_riesgo", [])
        asistentes = scout.get("asistentes_clave", [])
        portero = scout.get("portero_destacado", {}) or {}
        mercado_top = scout.get("mercado_jugador_top", {}) or {}
        scout_str = "\nSCOUT JUGADORES (Groq R2):"
        if goleadores:
            scout_str += "\n  🔥 Goleadores/valor: " + " | ".join(
                f"{g['jugador']} ({g['equipo']}) — {g['motivo']} [{g.get('mercado_sugerido','?')}]"
                for g in goleadores[:5]
            )
        if asistentes:
            scout_str += "\n  🎯 Asistentes: " + ", ".join(
                f"{a['jugador']} ({a['equipo']})" for a in asistentes[:3]
            )
        if portero.get("jugador"):
            scout_str += f"\n  🧤 Portero destacado: {portero['jugador']} ({portero.get('equipo','?')}) — {portero.get('motivo','')} [{portero.get('mercado','')}]"
        scout_str += f"\n  🎯 Lanzador penaltis local: {scout.get('lanzador_penaltis_local','?')}"
        scout_str += f"\n  🎯 Lanzador penaltis visit: {scout.get('lanzador_penaltis_visitante','?')}"
        _tiros_local = scout.get('tiros_a_puerta_lider_local') or {}
        _tiros_visit = scout.get('tiros_a_puerta_lider_visitante') or {}
        scout_str += f"\n  ⚡ Tiros lider local: {_tiros_local.get('jugador','?')} ({_tiros_local.get('tiros_partido','?')} tiros/partido)"
        scout_str += f"\n  ⚡ Tiros lider visit: {_tiros_visit.get('jugador','?')} ({_tiros_visit.get('tiros_partido','?')} tiros/partido)"
        if tarjeta_riesgo:
            scout_str += "\n  🟨 Riesgo tarjeta: " + " | ".join(
                f"{t['jugador']} ({t['equipo']}) — {t.get('motivo','')}"
                for t in tarjeta_riesgo[:4]
            )
        scout_str += f"\n  🔄 Corners local: {scout.get('corners_ejecutor_local','?')} | visit: {scout.get('corners_ejecutor_visitante','?')}"
        if scout.get("sorpresa_posible"):
            scout_str += f"\n  💡 Sorpresa posible: {scout['sorpresa_posible']}"
        if mercado_top.get("mercado"):
            scout_str += f"\n  ⭐ MERCADO TOP jugador: {mercado_top['mercado']} (cuota ~{mercado_top.get('cuota_estimada','?')}) — {mercado_top.get('motivo','')}"

    pred_str = ""
    if pred:
        pred_str = (
            f"\nSÍNTESIS PREDICTIVA (Groq R3):"
            f"\n  Resultado más probable: {pred.get('resultado','?')}"
            f"\n  Marcador exacto probable: {pred.get('marcador','?')}"
            f"\n  Señal de valor detectada: {pred.get('senal_valor','ninguna')}"
        )
        if pred.get("h2h_corners_promedio"):
            pred_str += f"\n  Media corners H2H: {pred['h2h_corners_promedio']}"

    groq_aviso = (
        f"\nNOTA: Forma viene de entrenamiento IA (confianza {groq_confianza}), NO tiempo real."
        if groq_usada and forma_desde_groq else ""
    )

    # V10: Bloque de contexto competicional
    comp_aviso_bloque = ""
    if comp_ctx.get("aviso_prompt"):
        tarj_adj  = comp_ctx["tarjetas_ajuste"]
        goles_adj = comp_ctx["goles_ajuste"]
        corn_adj  = comp_ctx["corners_ajuste"]
        fase      = comp_ctx.get("fase_detectada", "") or "Liga regular"
        nivel     = comp_ctx["nivel_alerta"]
        comp_aviso_bloque = (
            f"\n\n═══ CONTEXTO COMPETICIONAL — LEER OBLIGATORIAMENTE ═══"
            f"\nLIGA: {match['league']} | FASE: {fase} | ALERTA: {nivel}"
            f"\n{comp_ctx['aviso_prompt']}"
            f"\nAJUSTES NUMÉRICOS vs media habitual del árbitro:"
            f"\n  · Tarjetas esperadas: {tarj_adj:+.1f}"
            f"\n  · Goles esperados: {goles_adj:+.1f}"
            f"\n  · Corners esperados: {corn_adj:+.1f}"
            f"\nREGLA OBLIGATORIA: Si el árbitro promedia X tarjetas/partido,"
            f" en este contexto espera X+({tarj_adj:+.1f}) tarjetas reales."
            f" Usa ESE número ajustado para elegir el umbral del mercado."
            f"\nIGNORAR ESTE AJUSTE ES UN ERROR ESTADÍSTICO GRAVE."
        )

    scout_goleadores_lista = ", ".join(
        f"{g['jugador']} ({g['equipo']})" for g in scout.get("goleadores_calientes", [])
    ) if scout else "ninguno"
    def _fmt_tarjeta(t):
        base = f"{t['jugador']} ({t['equipo']})"
        extras = []
        if t.get("amarillas_por_partido") is not None:
            extras.append(f"{t['amarillas_por_partido']:.2f} amar/partido")
        if t.get("posicion_riesgo"):
            extras.append(t["posicion_riesgo"])
        if t.get("acumulacion_riesgo"):
            extras.append("⚠️acum")
        val = t.get("valor","")
        if val:
            extras.append(f"val={val}")
        return base + (f" [{', '.join(extras)}]" if extras else "")
    scout_tarjetas_lista = " | ".join(
        _fmt_tarjeta(t) for t in scout.get("jugadores_tarjeta_riesgo", [])
    ) if scout else "ninguno"
    scout_senal_tarjetas = scout.get("senal_tarjetas_totales", "sin_señal") if scout else "sin_señal"
    scout_portero = scout.get("portero_destacado", {}) or {} if scout else {}
    scout_mercado_top = scout.get("mercado_jugador_top", {}) or {} if scout else {}
    scout_sorpresa = scout.get("sorpresa_posible", "") if scout else ""

    prompt = f"""Eres un Trader Deportivo profesional. Has completado 3 rondas de análisis previo. Ahora genera el informe FINAL.

═══ DATOS VERIFICADOS (TIEMPO REAL) ═══
PARTIDO: {match['home']} vs {match['away']} — {match['league']}
Estado: {estado} | Marcador: {match.get('home_goals',0)}-{match.get('away_goals',0)}
Árbitro: {arbitro_final or "NO DISPONIBLE"} — {arb_hist_str} [{arb_hist_src}]
Alineación {match['home']}: {alin_local_str}
Alineación {match['away']}: {alin_visit_str}
Forma {match['home']}: {forma_local} | Forma {match['away']}: {forma_visit}
H2H: {h2h_str}{h2h_goles_str}
Bajas {match['home']}: {bajas_local_str}
Bajas {match['away']}: {bajas_visit_str}
{groq_aviso}{comp_aviso_bloque}

═══ MODELO CUANTITATIVO (Poisson/Dixon-Coles sobre datos reales) ═══
{format_quant_block(ctx.get('quant', {}))}
→ Estas probabilidades están CALCULADAS con datos reales, no estimadas. Son tu referencia
  numérica principal para goles, BTTS y 1X2. Tu papel es detectar VALOR frente a las cuotas
  y explicar el razonamiento, NO recalcular las probabilidades.

═══ ANÁLISIS PREVIO 3 RONDAS ═══
{patron_str}
{scout_str}
{pred_str}

═══ JUGADORES DETECTADOS POR EL SCOUT ═══
Goleadores/calientes: {scout_goleadores_lista}
Riesgo tarjeta: {scout_tarjetas_lista}
Señal tarjetas totales (scout): {scout_senal_tarjetas}
Portero destacado: {scout_portero.get('jugador','ninguno')} — {scout_portero.get('motivo','')}
Mercado top jugador: {scout_mercado_top.get('mercado','ninguno')} ({scout_mercado_top.get('jugador','')}, ~{scout_mercado_top.get('cuota_estimada','?')})
Sorpresa posible: {scout_sorpresa or 'ninguna'}

═══ REGLAS DE COHERENCIA NUMÉRICA — CRÍTICO, NO IGNORAR ═══
Árbitro — tarjetas: {arb_hist_str}
  → Si el árbitro promedia X tarjetas/partido, el lado correcto es: si promedia >= umbral, recomienda MÁS. Si promedia < umbral, recomienda MENOS. Pero sobretodo ten en cuenta si va ser un partido con mucha tension o suave para tomar la decision final.

Goles — media H2H: {h2h_goles_api if h2h_goles_api is not None else "no disponible"} | Goles esperados patrón: {patron.get('goles_esperados_total', 'no disponible')}
  → OBLIGATORIO: Analiza si es un partido propenso a muchos o pocos goles y si es uno de los dos extremos recomienda mas de o menos de x goles

Corners — esperados patrón: {patron.get('corners_esperados_total', 'no disponible')} | Ajuste competencia: {comp_ctx.get('corners_ajuste', 0):+f}
  → Corners esperados ajustados: {((patron.get('corners_esperados_total') or 0) + comp_ctx.get('corners_ajuste', 0)) if patron.get('corners_esperados_total') is not None else 'no disponible'}
  → Señal corners: {patron.get('senal_corners', 'sin_señal')} | Confianza estimación: {patron.get('corners_confianza', 'baja')}
  → REGLA: Recomienda mercado de corners SOLO si senal_corners="clara". Si es "dudosa" o "sin_señal", omite corners y sustituye por otro mercado con señal real. Una estimación de corners inventada por IA es peor que no tener mercado de corners.

BTTS — patrón: {patron.get('btts_probable', 'no disponible')}
  → OBLIGATORIO: si patrón es false/recomienda y el partido promete aburrido, recomienda BTTS no, si el patron es true y esta claro que van a meter muchos goles recomienda BTTS si.

Penaltis — árbitro: {f"{arb_penaltis:.2f}/partido" if arb_penaltis is not None else "no disponible"}
  → OBLIGATORIO: si probabilidad por partido < 0.2 → recomienda "Menos de 0.5 penaltis"
  → Si probabilidad por partido >= 0.2 → recomienda "Más de 0.5 penaltis"

INCOMPATIBILIDADES ENTRE LOS 8 MERCADOS — NUNCA incluyas a la vez:
  ❌ "BTTS: Sí" + "Menos de 2.5 goles"
  ❌ "BTTS: Sí" + "Menos de 1.5 goles"
  ❌ "Menos de X.5 goles" cuando la media esperada >= X
  ❌ "Más de X.5 tarjetas" cuando el árbitro promedia menos de X tarjetas/partido
  ❌ "Menos de X.5 tarjetas" cuando el árbitro promedia más de X tarjetas/partido
  ❌ "Tarjeta amarilla para [jugador]" si ese jugador tiene amarillas_por_partido < 0.25 o valor "bajo"
  ❌ Cualquier mercado de tarjetas si senal_tarjetas_totales = "sin_señal" Y árbitro_tarjetas_partido < 4.0

═══ REGLAS ESPECÍFICAS DE TARJETAS — LEER CON ATENCIÓN ═══
Para recomendar mercados de TARJETAS TOTALES (Más/Menos X.5):
  ✅ SOLO recomienda "Más de X.5 tarjetas" si SE CUMPLEN AL MENOS 2 DE ESTAS 3 condiciones:
      1. árbitro_tarjetas_ajustado >= X+1.0 (árbitro supera claramente el umbral)
      2. Hay >=2 jugadores con valor "alto" en jugadores_tarjeta_riesgo
      3. El partido tiene alta tensión: eliminatoria vuelta, derby, lucha por título/descenso
  ✅ SOLO recomienda "Menos de X.5 tarjetas" si SE CUMPLEN AL MENOS 2 DE ESTAS 3 condiciones:
      1. árbitro_tarjetas_ajustado <= X-1.0 (árbitro está claramente por debajo)
      2. Contexto de baja tensión: partido sin nada en juego, fase de grupo sin presión
      3. senal_tarjetas_totales = "sin_señal" en el scout

Para recomendar "Tarjeta amarilla para [jugador]" (apuesta a jugador concreto):
  ✅ Solo si: amarillas_por_partido >= 0.35 Y valor = "alto" Y árbitro no es "permisivo"
  ✅ Preferir jugadores en posicion_riesgo: "central", "pivote", "lateral" sobre atacantes
  ✅ Si acumulacion_riesgo=true, el jugador PODRÍA jugar más cauto — penaliza ligeramente

⚡ REGLA DE ORO: Si no tienes 2 condiciones claras para tarjetas totales y ningún jugador con señal alta, NO incluyas mercado de tarjetas. Es mejor poner otro mercado con señal real que forzar uno de tarjetas que probablemente falle.
  ❌ "Más de X.5 corners" cuando los corners esperados ajustados < X
  ❌ "Menos de X.5 corners" cuando los corners esperados ajustados >= X
  ❌ Cualquier mercado de corners si senal_corners = "sin_señal" o corners_confianza = "baja"
  ❌ "Más de 10.5 corners" si corners_confianza != "alta" (umbral muy alto, requiere datos reales)

═══ INSTRUCCIONES OBLIGATORIAS ═══
1. PRIMERO - EVALUACIÓN DE CATEGORÍAS: Antes de seleccionar líneas específicas, evalúa brevemente qué categorías de mercado (goles, corners, tarjetas, BTTS, resultado, penaltis, tiros, paradas, tiempo) presentan valor basado en el análisis. Menciona 2-3 categorías que consideras más prometedoras.
2. SEGUNDO - SELECCIÓN DE LÍNEAS: Para las categorías prometedoras identificadas, selecciona las líneas específicas más coherentes.
3. LECTURA TÁCTICA: 5-6 frases. Menciona el árbitro, el patrón esperado y AL MENOS 2 jugadores concretos del scout.
4. MERCADOS: Genera EXACTAMENTE 8 mercados con al menos: 1 jugador individual, 1 corners, 1 resultado/handicap.
   TARJETAS — INCLUIR SOLO SI HAY SEÑAL REAL (ver reglas abajo). Si no hay señal, sustituye por otro mercado de valor.
5. JUGADORES A VIGILAR: Lista TODOS los jugadores detectados por el scout con señal real. Mínimo 3.
6. Marca con "(orientativo)" cualquier dato de IA.
7. CONFIANZA GENERAL — REGLA OBLIGATORIA:
   - Si groq_confianza="{groq_confianza}" Y árbitro/alineaciones/forma NO son todos ✅ → DEBES poner "Baja" o "Media". NO puedes poner "Alta".
   - Solo puedes poner "Alta" si groq_confianza es "alta" O si todos los datos críticos (árbitro, alineaciones, forma) vienen de ✅ API/scraping real.
   - Esta regla es NO NEGOCIABLE. Incumplirla es un error grave de calibración.

NOMBRES DE MERCADOS EXACTOS:
GOLES: "Más/Menos de X.5 goles" | RESULTADO: "Victoria local" | "Victoria visitante" | "Empate"
BTTS: "BTTS: Sí" | "BTTS: No" | HÁNDICAP: "Hándicap asiático -X local"
TARJETAS: "Más de X.5 tarjetas" | "Menos de X.5 tarjetas" | "Tarjeta amarilla para [jugador]"
CORNERS: "Más de X.5 corners" | JUGADOR: "Anytime scorer: [jugador]" | "Primer gol: [jugador]"
TIROS: "Tiros a puerta totales Más de X.5" | PARADAS: "Paradas [portero] Más de X.5"
TIEMPO: "Primer gol antes del min 30" | EXACTO: "Marcador exacto [X-Y]" | "Penalti en el partido: Sí"

═══ FORMATO DE SALIDA (OBLIGATORIO, EXACTO) ═══

🧠 LECTURA TÁCTICA:
[5-6 frases directas.]

🎯 MERCADOS RECOMENDADOS:
1️⃣ [Mercado exacto] -> Cuota aprox [X.XX] | Prob: [YY.Y%] | Base: [dato real específico]
2️⃣ [Mercado exacto] -> Cuota aprox [X.XX] | Prob: [YY.Y%] | Base: [dato real específico]
3️⃣ [Mercado exacto] -> Cuota aprox [X.XX] | Prob: [YY.Y%] | Base: [dato real específico]
4️⃣ [Mercado exacto] -> Cuota aprox [X.XX] | Prob: [YY.Y%] | Base: [dato real específico]
5️⃣ [Mercado exacto] -> Cuota aprox [X.XX] | Prob: [YY.Y%] | Base: [dato real específico]
6️⃣ [Mercado exacto] -> Cuota aprox [X.XX] | Prob: [YY.Y%] | Base: [dato real específico]
7️⃣ [Mercado exacto] -> Cuota aprox [X.XX] | Prob: [YY.Y%] | Base: [dato real específico]
8️⃣ [Mercado exacto] -> Cuota aprox [X.XX] | Prob: [YY.Y%] | Base: [dato real específico]

IMPORTANTE: Prob [YY.Y%] = tu estimación REAL de que ese evento ocurra, basada en los datos del partido (forma, H2H, árbitro, patrón). NO es 1/cuota. La probabilidad puede ser mayor o menor que la implícita en la cuota — eso es precisamente donde está el valor. Por ejemplo: si el árbitro promedia 6 tarjetas/partido y el scout detecta jugadores de riesgo, tu prob real para "Más de 3.5 tarjetas" puede ser 65% aunque la cuota sea 2.50.

🔍 JUGADORES A VIGILAR:
[Para cada jugador: "▸ NOMBRE (Equipo): [estadística]. Mercado: [nombre exacto] — Cuota aprox X.XX | Prob: YY.Y%"]

⚠️ Confianza General: [Alta/Media/Baja] — [razón en 1 frase]"""

    resp = await ask_groq(prompt, temperature=0.72)  # temperatura alta = análisis específico por partido, evita respuestas genéricas
    if not resp:
        return "❌ Groq está en pausa temporal. Inténtalo en un par de minutos."

    # ── Opcional: Segunda pasada de análisis para bajo confianza o validación débil ────────────────────────
    # Extraer nivel de confianza del trader para decidir si hacer refinamiento
    confianza_line = ""
    for line in resp.split("\n"):
        if "Confianza General:" in line:
            confianza_line = line
            break

    # Si la confianza es media o baja, o si no encontramos línea de confianza, considerar refinamiento
    necesita_refinamiento = (
        "Confianza General: Media" in confianza_line or
        "Confianza General: Baja" in confianza_line or
        not confianza_line
    )

    if necesita_refinamiento:
        # Crear un prompt de refinamiento más enfocado basado en el análisis inicial
        prompt_refinamiento = f"""Basándote en el análisis anterior, proporciona una versión mejorada y más concisa que corrija cualquier incoherencia y enfoque en las oportunidades más sólidas.

ANÁLISIS INICIAL:
{resp}

INSTRUCCIONES DE REFINAMIENTO:
1. Mantén los hechos verificados (estadios, árbitros, alineaciones, forma, H2H, bajas)
2. Mejora la LECTURA TÁCTICA siendo más específica y basada en los datos
3. Revisa los 8 MERCADOS para eliminar cualquier incoherencia evidente
4. Asegúrate de que JUGADORES A VIGILAR incluya al menos 3 jugadores con señales claras
5. Si es posible, aumenta la confianza identificando oportunidades más sólidas
6. Usa el mismo formato exacto que el análisis original

ANÁLISIS REFINADO:"""

        resp_refinado = await ask_groq(prompt_refinamiento, expect_json=False, temperature=0.25)
        if resp_refinado and len(resp_refinado) > 35:  # Solo usar si parece válido
            resp = resp_refinado
            logger.info("Usando análisis refinado para mejorar confianza")

    # ── Ronda 4: Megacombinada ───────────────────────────────────────────────
    goles_esp_str = str(patron.get('goles_esperados_total', '?'))
    btts_pat_str  = str(patron.get('btts_probable', '?'))
    arb_tarj_str  = f"{arb_tarjetas:.1f}" if arb_tarjetas is not None else "desconocido"

    prompt_mega = f"""Eres un Trader Deportivo experto en apuestas combinadas. Tienes el análisis completo de este partido:

{resp.strip()}

Datos clave del partido:
- Árbitro: {arbitro_final or 'desconocido'} — {arb_hist_str}
- Forma local: {forma_local} | Forma visitante: {forma_visit}
- H2H: {h2h_str}{h2h_goles_str}
- Patrón: goles esperados={patron.get('goles_esperados_total','?')}, corners={patron.get('corners_esperados_total','?')}, BTTS={patron.get('btts_probable','?')}, tarjetas esperadas={patron.get('tarjetas_esperadas','?')}
- Corners ajustados: {((patron.get('corners_esperados_total') or 0) + comp_ctx.get('corners_ajuste', 0)) if patron.get('corners_esperados_total') is not None else 'no disponible'}
- Señal de valor: {pred.get('senal_valor','ninguna') if pred else 'ninguna'}
- CONTEXTO COMPETICIONAL (V10): Fase={comp_ctx.get('fase_detectada','Liga regular')} | Ajuste tarjetas={comp_ctx['tarjetas_ajuste']:+.1f} | Ajuste goles={comp_ctx['goles_ajuste']:+.1f} | Ajuste corners={comp_ctx['corners_ajuste']:+.1f}
  → Árbitro media ajustada real (incluyendo fase): {f"{arb_tarjetas_ajustado:.1f}" if arb_tarjetas_ajustado is not None else "desconocida"} tarjetas/partido
  → Corners esperados ajustados: {((patron.get('corners_esperados_total') or 0) + comp_ctx.get('corners_ajuste', 0)) if patron.get('corners_esperados_total') is not None else 'no disponible'}

Tu tarea: construir LA MEGACOMBINADA ÓPTIMA del partido.

REGLAS ESTRICTAS:
1. Elige entre 2 y 4 selecciones (no más de 4).
2. Selecciona SOLO picks con alta coherencia entre sí y respaldados por datos concretos.
3. Cada pick DEBE tener base en un dato real específico del análisis (estadística de árbitro, H2H, forma, scout jugador...).
4. Calcula la cuota combinada multiplicando las cuotas individuales.
5. Si la cuota combinada supera 10.0 es demasiado arriesgada.
6. Prioriza picks con cuota entre 1.40 y 3.00.
7. ⭐ DIVERSIDAD OBLIGATORIA: NO puedes poner más de 1 pick de goles (ni BTTS+más de 2.5 juntos, son casi lo mismo). Intenta combinar mercados de CATEGORÍAS DISTINTAS: por ejemplo un pick de goles/BTTS + un pick de tarjetas o corners + un pick de jugador individual. Así la combinada tiene picks independientes entre sí.
8. ⭐ VALOR REAL: La prob estimada de cada pick es tu estimación real de que ocurra según los datos, NO la inversa de la cuota (1/cuota). Busca picks donde los datos del partido te dan más confianza que la cuota de mercado. Si todos los picks tienen prob similar a 1/cuota, no estás aportando valor — analiza más en profundidad.
9. ⭐ FILTRO MÍNIMO: Solo incluye picks con prob estimada real >35%. Si no encuentras 2 picks con prob >35% respaldados por datos, indica "SIN COMBINADA SEGURA" y explica por qué el partido no ofrece valor claro.

INCOMPATIBILIDADES ABSOLUTAS — NUNCA combines:
❌ "BTTS: Sí" + "Menos de 2.5 goles"
❌ "BTTS: Sí" + "Menos de 1.5 goles"
❌ "BTTS: No" + "Más de 2.5 goles"
❌ "Más de X.5 tarjetas" si el árbitro promedia menos de X tarjetas/partido (dato: {arb_tarj_str}/partido)
❌ "Menos de X.5 tarjetas" si el árbitro promedia más de X tarjetas/partido (dato: {arb_tarj_str}/partido)
❌ "Más de X.5 corners" si los corners esperados ajustados < X
❌ "Menos de X.5 corners" si los corners esperados ajustados >= X
❌ Cualquier mercado de goles que contradiga el patrón esperado de {goles_esp_str} goles
❌ BTTS: Sí si btts_probable={btts_pat_str} indica false
❌ BTTS: No si btts_probable={btts_pat_str} indica true
❌ NUNCA incluyas un pick con probabilidad <=35%
❌ "BTTS: Sí" + "Más de 2.5 goles" en la MISMA combinada — son picks REDUNDANTES y altamente correlacionados: si falla uno, cae el otro. Elige solo UNO de los dos.

FORMATO DE SALIDA (EXACTO):

━━━━━━━━━━━━━━━━━━━━━━━━
🎰 MEGACOMBINADA RECOMENDADA
━━━━━━━━━━━━━━━━━━━━━━━━
✅ Pick 1: [mercado exacto] — Cuota ~X.XX | Prob. estimada: XX% | [1 frase base de datos]
✅ Pick 2: [mercado exacto] — Cuota ~X.XX | Prob. estimada: XX% | [1 frase base de datos]
[Pick 3 , 4 y 5 opcionales solo si son 100% compatibles y tienen probabilidad >35%]

💰 CUOTA COMBINADA ESTIMADA: ~X.XX
📊 PROB. COMBINADA ESTIMADA: ~XX%
🎯 CONFIANZA: [Alta/Media/Baja]
⚠️ RIESGO: [1 frase explicando el principal factor de riesgo]"""

    resp_mega = await ask_groq(prompt_mega)

    # Validar coherencia de la megacombinada
    if resp_mega:
        resp_mega = _validar_megacombinada(
            texto_mega=resp_mega,
            arb_tarjetas=arb_tarjetas,
            patron=patron,
            h2h_goles_api=h2h_goles_api,
            comp_ctx=comp_ctx,
        )

    # Footer fijo
    fuentes_reales = ctx.get("fuentes_consultadas", [])
    groq_conf = ctx.get("groq_confianza", "")
    groq_nota = f" (Groq confianza: {groq_conf})" if groq_conf and "Groq/conocimiento" in fuentes_reales else ""
    footer = (
        f"\n\n_📡 Fuentes scraping: {', '.join(f for f in fuentes_reales if f != 'Groq/conocimiento') or 'ninguna (bloqueadas 403)'}_\n"
        f"_🤖 Enriquecimiento IA: {'Groq/conocimiento' + groq_nota if 'Groq/conocimiento' in fuentes_reales else 'no usado'}_\n"
        f"_🕐 {datetime.fromtimestamp(ctx.get('timestamp', time.time())).strftime('%H:%M')}_"
    )

    mega_block = f"\n\n{resp_mega.strip()}" if resp_mega else ""
    resultado = f"{cabecera}\n\n{resp.strip()}{mega_block}\n{footer}"

    # ── Validación cruzada determinista (sin coste de API) ─────────────────────
    resultado = _validar_coherencia(
        texto=resultado,
        ctx=ctx,
        arb_tarjetas=arb_tarjetas,
        arb_penaltis=arb_penaltis,
        h2h_goles_api=h2h_goles_api,
        patron=patron,
        comp_ctx=comp_ctx,
    )

    # Validación: Groq solo detecta problemas, NUNCA reescribe el resultado completo
    goles_esperados_patron = patron.get('goles_esperados_total')
    corners_esperados_patron = patron.get('corners_esperados_total')
    btts_patron = patron.get('btts_probable')
    tarjetas_esperadas_patron = patron.get('tarjetas_esperadas')

    validation_prompt = f"""Eres un validador lógico de pronósticos deportivos. Tu única tarea es detectar CONTRADICCIONES REALES (no comentarios ni aclaraciones).

REGLAS DE CONTRADICCIÓN (solo estas, nada más):
- "BTTS: Sí" junto con "Menos de 2.5 goles" → CONTRADICCIÓN
- "BTTS: Sí" junto con "Menos de 1.5 goles" → CONTRADICCIÓN  
- "BTTS: No" junto con "Más de 2.5 goles" → CONTRADICCIÓN
- "Más de X.5 tarjetas" cuando árbitro_tarjetas_ajustado < X → CONTRADICCIÓN (árbitro no llega al umbral)
- "Menos de X.5 tarjetas" cuando árbitro_tarjetas_ajustado > X+1 → CONTRADICCIÓN (árbitro supera el umbral)
- "Más de X.5 goles" cuando goles_esperados < X-0.5 → CONTRADICCIÓN
- "Menos de X.5 goles" cuando goles_esperados > X+0.5 → CONTRADICCIÓN

EJEMPLOS DE LO QUE NO ES CONTRADICCIÓN (NO reportes esto):
✅ "Más de 3.5 tarjetas" cuando árbitro_tarjetas_ajustado = 5.0 o 6.0 → CORRECTO (árbitro supera el umbral)
✅ "Más de 3.5 tarjetas" cuando árbitro_tarjetas_ajustado = 4.0 → CORRECTO (supera 3.5)
✅ "Más de 2.5 goles" cuando goles_esperados = 2.5 → zona límite, NO es contradicción
✅ Que la media del árbitro (6.0) sea mayor que el umbral del mercado (3.5) → esto APOYA el mercado, NO es contradicción

DATOS DE REFERENCIA:
- Goles esperados patrón: {goles_esperados_patron}
- BTTS patrón: {btts_patron}
- Tarjetas árbitro reales ajustadas: {arb_tarjetas_ajustado} (ya incluye ajuste competicional)
- Tarjetas patrón: {tarjetas_esperadas_patron}

MERCADOS A VALIDAR (extrae solo los mercados recomendados del análisis):
{resultado[resultado.find("🎯 MERCADOS"):resultado.find("🔍 JUGADORES")] if "🎯 MERCADOS" in resultado else resultado[:800]}

INSTRUCCIÓN FINAL:
- Si NO hay contradicciones reales según las reglas de arriba → responde SOLO: VALID
- Si hay contradicciones → lista SOLO las contradicciones reales (máx 2 líneas, formato ⚠️ ...)
- NO comentes diferencias normales entre datos. NO expliques nada. NO copies el análisis.
- RECUERDA: que el árbitro promedia MÁS tarjetas que el umbral del mercado es CONSISTENTE, no contradictorio.

Respuesta:"""

    validation_response = await ask_groq(validation_prompt, max_retries=6, temperature=0.1)
    if (validation_response
            and validation_response.strip().upper() != "VALID"
            and len(validation_response) > 10
            and len(validation_response) < 800):   # respuesta corta = lista de avisos, no una reescritura
        avisos_groq = validation_response.strip()
        if "_📡 Fuentes" in resultado:
            resultado = resultado.replace(
                "_📡 Fuentes",
                f"\n⚠️ *Revisión automática:*\n{avisos_groq}\n\n_📡 Fuentes"
            )
        else:
            resultado += f"\n\n⚠️ *Revisión automática:*\n{avisos_groq}"
        logger.info("Validación Groq: avisos añadidos al footer.")

    # ── V11: bloque cuantitativo visible + correcciones CoVe en el mensaje ────
    quant_ctx = ctx.get("quant", {})
    if quant_ctx.get("disponible"):
        bloque_quant = (
            f"\n\n📊 *Modelo cuantitativo* _( {quant_ctx['fuente']} )_\n"
            f"xG: {quant_ctx['xg_local']} - {quant_ctx['xg_visitante']} | "
            f"1X2: {quant_ctx['p_local']:.0%}/{quant_ctx['p_empate']:.0%}/{quant_ctx['p_visitante']:.0%} | "
            f"O2.5: {quant_ctx['p_over_2_5']:.0%} | BTTS: {quant_ctx['p_btts']:.0%}"
        )
        resultado += bloque_quant

    cove_notas = ctx.get("cove_correcciones", [])
    if cove_notas:
        resultado += "\n🔎 _Verificación automática: " + "; ".join(cove_notas[:3]) + "_"

    # ── V11 §2.5: registrar predicciones para backtesting/calibración ─────────
    fixture_id = match.get("fixture_id", 0)
    if fixture_id and quant_ctx.get("disponible"):
        traza = {"patron": ctx.get("groq_patron", {}), "quant": quant_ctx,
                 "cove": cove_notas, "comp_ctx": comp_ctx.get("fase_detectada", "")}
        conf_txt = str((ctx.get("groq_patron") or {}).get("confianza", ""))
        kickoff = match.get("date_iso", "")
        probs_1x2 = {"home": quant_ctx["p_local"], "draw": quant_ctx["p_empate"], "away": quant_ctx["p_visitante"]}
        fav = max(probs_1x2, key=probs_1x2.get)
        db_log_prediction(fixture_id, match["home"], match["away"], match["league"], kickoff,
                          "1X2", fav, probs_1x2[fav], conf_txt, traza)
        db_log_prediction(fixture_id, match["home"], match["away"], match["league"], kickoff,
                          "over_2_5", "si" if quant_ctx["p_over_2_5"] >= 0.5 else "no",
                          quant_ctx["p_over_2_5"] if quant_ctx["p_over_2_5"] >= 0.5 else 1 - quant_ctx["p_over_2_5"],
                          conf_txt, {})
        db_log_prediction(fixture_id, match["home"], match["away"], match["league"], kickoff,
                          "btts", "si" if quant_ctx["p_btts"] >= 0.5 else "no",
                          quant_ctx["p_btts"] if quant_ctx["p_btts"] >= 0.5 else 1 - quant_ctx["p_btts"],
                          conf_txt, {})

    # ── V11: aviso de juego responsable en cada señal ─────────────────────────
    resultado += DISCLAIMER

    return _sanitize_markdown(_limpiar_salida(resultado))

def _limpiar_salida(texto: str) -> str:
    # Corrige patrones de dobles ".5"
    texto = re.sub(r'(\d+)\.5\.5', r'\1.5', texto)
    # Corrige palabras duplicadas comunes
    texto = texto.replace('tarjetass', 'tarjetas')
    texto = texto.replace('goalss', 'goals')
    texto = texto.replace('cornerss', 'corners')
    return texto

# ═══════════════════════════════════════════════════════════
#  UTILIDADES TELEGRAM
# ═══════════════════════════════════════════════════════════
def split_message(text: str, limit: int = TELEGRAM_LIMIT) -> list:
    if len(text) <= limit: return [text]
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        cut = text.rfind("\n", 0, limit)
        if cut == -1: cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks

def _inyectar_probabilidades(text: str) -> str:
    import re as _re

    def _add_prob_mercado(m):
        pre   = m.group(1)
        cuota = m.group(2)
        sep   = m.group(3)
        resto = m.group(4)
        try:
            prob = round(100.0 / float(cuota), 1)
            prob_str = f"{prob}%"
        except (ValueError, ZeroDivisionError):
            prob_str = "?%"
        if "Prob" in pre:
            return m.group(0)
        return f"{pre} | Prob: {prob_str} | {sep}{resto}"

    text = _re.sub(
        r"(->\s*Cuota\s+aprox\s+([\d.]+))\s*\|\s*([Bb]ase\s*:)(.*)",
        _add_prob_mercado,
        text
    )

    def _add_prob_jugador(m):
        pre   = m.group(1)
        cuota = m.group(2)
        post  = m.group(3)
        try:
            prob = round(100.0 / float(cuota), 1)
            prob_str = f"{prob}%"
        except (ValueError, ZeroDivisionError):
            prob_str = "?%"
        if "Prob" in (pre + post):
            return m.group(0)
        return f"{pre} | Prob: {prob_str}{post}"

    text = _re.sub(
        r"(—\s*Cuota\s+aprox\s+([\d.]+))((?!\s*\|?\s*Prob)[^\n]*)",
        _add_prob_jugador,
        text
    )
    return text


def _sanitize_markdown(text: str) -> str:
    text = re.sub(r'(\d)\*', r'\1', text)
    text = re.sub(r'(?<!\s)_(?!\s)', r' ', text)
    text = re.sub(r'(?<!`)`(?!`)', '', text)
    return text


# ═══════════════════════════════════════════════════════════
#  VALIDADOR DE COHERENCIA
# ═══════════════════════════════════════════════════════════

def _validar_coherencia(texto: str, ctx: dict, arb_tarjetas: Optional[float],
                        arb_penaltis: Optional[float], h2h_goles_api: Optional[float],
                        patron: dict, comp_ctx: Optional[dict] = None) -> str:
    alertas = []

    tarjeta_menos = re.findall(r'[Mm]enos de (\d+(?:\.\d+)?)\.?5? tarjeta', texto)
    tarjeta_mas   = re.findall(r'[Mm][áa]s de (\d+(?:\.\d+)?)\.?5? tarjeta', texto)
    goles_menos   = re.findall(r'[Mm]enos de (\d+(?:\.\d+)?)\.?5? goles?', texto)
    goles_mas     = re.findall(r'[Mm][áa]s de (\d+(?:\.\d+)?)\.?5? goles?', texto)
    corners_menos = re.findall(r'[Mm]enos de (\d+(?:\.\d+)?)\.?5? corners?', texto)
    corners_mas   = re.findall(r'[Mm][áa]s de (\d+(?:\.\d+)?)\.?5? corners?', texto)
    btts_si  = bool(re.search(r'BTTS[:\s]+S[íi]', texto, re.IGNORECASE))
    btts_no  = bool(re.search(r'BTTS[:\s]+No', texto, re.IGNORECASE))
    penalti_si = bool(re.search(r'[Pp]enalti en el partido[:\s]+S[íi]', texto))

    if arb_tarjetas is not None:
        for val_str in tarjeta_menos:
            try:
                umbral = float(val_str) + 0.5
                if arb_tarjetas >= umbral:
                    alertas.append(
                        f"⚠️ CONTRADICCIÓN DETECTADA: Se recomienda 'Menos de {umbral:.0f} tarjetas' "
                        f"pero el árbitro promedia {arb_tarjetas:.1f} tarjetas/partido "
                        f"(igual o por encima del umbral). Valora 'Más de {umbral:.0f} tarjetas' en su lugar."
                    )
            except ValueError:
                pass

        for val_str in tarjeta_mas:
            try:
                umbral = float(val_str) + 0.5
                if arb_tarjetas < umbral - 1.0:
                    alertas.append(
                        f"⚠️ CONTRADICCIÓN DETECTADA: Se recomienda 'Más de {umbral:.0f} tarjetas' "
                        f"pero el árbitro promedia {arb_tarjetas:.1f} tarjetas/partido. "
                        f"Valora 'Menos de {umbral:.0f} tarjetas' en su lugar."
                    )
            except ValueError:
                pass

    goles_ref = None
    if h2h_goles_api is not None:
        goles_ref = h2h_goles_api
    elif patron.get("goles_esperados_total") is not None:
        try:
            goles_ref = float(patron["goles_esperados_total"])
        except (TypeError, ValueError):
            pass

    if goles_ref is not None:
        for val_str in goles_menos:
            try:
                umbral = float(val_str) + 0.5
                if goles_ref >= umbral:
                    alertas.append(
                        f"⚠️ CONTRADICCIÓN DETECTADA: Se recomienda 'Menos de {umbral:.0f} goles' "
                        f"pero la referencia histórica/esperada es {goles_ref:.1f} goles/partido. Revisa."
                    )
            except ValueError:
                pass

        for val_str in goles_mas:
            try:
                umbral = float(val_str) + 0.5
                if goles_ref < umbral - 1.0:
                    patron_buscar   = re.compile(
                        rf'[Mm][áa]s de {re.escape(val_str)}[\.,]?5? goles?', re.IGNORECASE
                    )
                    texto_corregido = patron_buscar.sub(f"Menos de {val_str}.5 goles", texto)
                    if texto_corregido != texto:
                        texto = texto_corregido
                        logger.warning(
                            f"Auto-corrección goles: H2H={goles_ref:.1f} < umbral {umbral:.1f} — corregido"
                        )
                        alertas.append(
                            f"⚠️ AUTO-CORRECCIÓN APLICADA: 'Más de {umbral:.1f} goles' → 'Menos de {umbral:.1f} goles' "
                            f"(la media histórica/esperada es solo {goles_ref:.1f})."
                        )
                    else:
                        alertas.append(
                            f"⚠️ CONTRADICCIÓN DETECTADA: Se recomienda 'Más de {umbral:.1f} goles' "
                            f"pero la referencia histórica/esperada es solo {goles_ref:.1f} goles/partido. Revisa."
                        )
            except ValueError:
                pass

    corners_ref = None
    try:
        cv = patron.get("corners_esperados_total")
        if cv is not None:
            corners_ref = float(cv)
    except (TypeError, ValueError):
        pass

    if corners_ref is not None:
        for val_str in corners_menos:
            try:
                umbral = float(val_str) + 0.5
                if corners_ref >= umbral:
                    alertas.append(
                        f"⚠️ CONTRADICCIÓN DETECTADA: 'Menos de {umbral:.0f} corners' pero el modelo espera ~{corners_ref:.0f}. Revisa."
                    )
            except ValueError:
                pass
        for val_str in corners_mas:
            try:
                umbral = float(val_str) + 0.5
                if corners_ref < umbral - 2.0:
                    alertas.append(
                        f"⚠️ CONTRADICCIÓN DETECTADA: 'Más de {umbral:.0f} corners' pero el modelo solo espera ~{corners_ref:.0f}. Revisa."
                    )
            except ValueError:
                pass

    btts_patron = patron.get("btts_probable")
    if btts_patron is not None:
        if btts_si and btts_patron is False:
            alertas.append(
                "⚠️ CONTRADICCIÓN DETECTADA: Se recomienda 'BTTS: Sí' pero el análisis indica que BTTS es poco probable."
            )
        if btts_no and btts_patron is True:
            alertas.append(
                "⚠️ CONTRADICCIÓN DETECTADA: Se recomienda 'BTTS: No' pero el análisis indica que BTTS es probable."
            )

    if penalti_si and arb_penaltis is not None and arb_penaltis < 0.10:
        alertas.append(
            f"⚠️ CONTRADICCIÓN DETECTADA: 'Penalti en el partido: Sí' pero el árbitro concede solo {arb_penaltis:.2f} penaltis/partido."
        )

    if not alertas:
        return texto

    bloque_alertas = (
        "\n\n⚠️ *VALIDACIÓN DE COHERENCIA — REVISAR ANTES DE APOSTAR:*\n"
        + "\n".join(f"  {a}" for a in alertas)
    )

    if "_📡 Fuentes" in texto:
        texto = texto.replace("_📡 Fuentes", bloque_alertas + "\n\n_📡 Fuentes")
    else:
        texto += bloque_alertas

    logger.warning(f"Validador coherencia: {len(alertas)} contradicción(es) detectada(s).")
    return texto

def _validar_megacombinada(texto_mega: str, arb_tarjetas: Optional[float],
                            patron: dict, h2h_goles_api: Optional[float],
                            comp_ctx: Optional[dict] = None) -> str:
    if not texto_mega:
        return texto_mega

    lines = texto_mega.split("\n")
    picks_text = " ".join(l for l in lines if l.strip().startswith("✅"))

    incompatibilidades = []    # errores reales que NO se pudieron auto-corregir
    autocorrections = []       # correcciones pendientes de aplicar
    avisos_autocorregidos = [] # correcciones ya aplicadas (solo informativas, no son errores)

    # ── FILTRO DE PROBABILIDAD: eliminar físicamente picks con prob <=35% ────
    lines_filtradas = []
    picks_eliminados = []
    for line in lines:
        if line.strip().startswith("✅"):
            m_prob = re.search(r'[Pp]rob\.\s*estimada[:\s]+(\d+(?:\.\d+)?)%', line)
            if m_prob:
                try:
                    prob = float(m_prob.group(1))
                    if prob <= 35.0:
                        picks_eliminados.append(
                            f"⚠️ PICK ELIMINADO (prob {prob:.1f}% ≤35%): {line.strip()}"
                        )
                        continue  # no añadir esta línea
                except ValueError:
                    pass
        lines_filtradas.append(line)

    if picks_eliminados:
        texto_mega = "\n".join(lines_filtradas)
        avisos_autocorregidos.extend(picks_eliminados)
        logger.warning(f"Megacombinada: {len(picks_eliminados)} pick(s) eliminado(s) por prob ≤35%.")

    # ── GUARDIA: si quedan menos de 2 picks, la combinada no es válida ────────
    picks_restantes = [l for l in lines_filtradas if l.strip().startswith("✅")]
    if len(picks_restantes) < 2:
        aviso_sin_combinada = (
            "\n⚠️ *COMBINADA NO DISPONIBLE*: Tras el filtro de calidad, no hay suficientes picks "
            "con probabilidad real >35% respaldados por datos. Es mejor no combinar en este partido."
        )
        texto_mega = texto_mega + aviso_sin_combinada
        logger.warning("Megacombinada: menos de 2 picks válidos — combinada marcada como no disponible.")

    # Re-parsear picks_text tras el filtrado
    lines = texto_mega.split("\n")
    picks_text = " ".join(l for l in lines if l.strip().startswith("✅"))

    # Comprobar prob combinada — si supera 10.0 de cuota o la prob combinada < 10% es demasiado arriesgada
    cuota_comb = re.search(r'CUOTA COMBINADA ESTIMADA[:\s~]+(\d+(?:\.\d+)?)', texto_mega)
    prob_comb  = re.search(r'PROB\. COMBINADA ESTIMADA[:\s~]+(\d+(?:\.\d+)?)%', texto_mega)
    if cuota_comb:
        try:
            if float(cuota_comb.group(1)) > 50.0:
                incompatibilidades.append(
                    f"❌ CUOTA COMBINADA DEMASIADO ALTA: {cuota_comb.group(1)} (máximo recomendado: 50.0). Reduce el número de picks."
                )
        except ValueError:
            pass
        except ValueError:
            pass

    btts_si  = bool(re.search(r'BTTS[:\s]+S[íi]', picks_text, re.IGNORECASE))
    btts_no  = bool(re.search(r'BTTS[:\s]+No', picks_text, re.IGNORECASE))
    goles_menos_vals = re.findall(r'[Mm]enos de (\d+(?:\.\d+)?)[\.,]?5? goles?', picks_text)
    goles_mas_vals   = re.findall(r'[Mm][áa]s de (\d+(?:\.\d+)?)[\.,]?5? goles?', picks_text)

    for v in goles_menos_vals:
        try:
            umbral = float(v) + 0.5
            if btts_si and umbral <= 2.5:
                incompatibilidades.append(
                    f"❌ INCOMPATIBLE: 'BTTS: Sí' + 'Menos de {umbral:.1f} goles' — Elimina uno de los dos picks."
                )
        except ValueError:
            pass

    # Detectar redundancia BTTS: Sí + Más de 2.5 goles
    for v in goles_mas_vals:
        try:
            umbral = float(v) + 0.5
            if btts_si and abs(umbral - 2.5) < 0.1:
                avisos_autocorregidos.append(
                    "⚠️ AVISO REDUNDANCIA: 'BTTS: Sí' + 'Más de 2.5 goles' en la misma combinada — son picks "
                    "altamente correlacionados. Si el partido acaba 0-0 o 1-0 ambos caen juntos. "
                    "Considera eliminar uno para mejorar la independencia de la combinada."
                )
        except ValueError:
            pass

    for v in goles_mas_vals:
        try:
            umbral = float(v) + 0.5
            if btts_no and umbral >= 2.5:
                incompatibilidades.append(
                    f"❌ INCOMPATIBLE: 'BTTS: No' + 'Más de {umbral:.1f} goles' — combinación de muy baja probabilidad."
                )
        except ValueError:
            pass

    # Validate and auto-correct tarjetas markets
    if arb_tarjetas is not None:
        tarj_menos_vals = re.findall(r'[Mm]enos de (\d+(?:\.\d+)?)[\.,]?5? tarjeta', picks_text)
        tarj_mas_vals   = re.findall(r'[Mm][áa]s de (\d+(?:\.\d+)?)[\.,]?5? tarjeta', picks_text)
        for v in tarj_menos_vals:
            try:
                umbral = float(v) + 0.5
                if arb_tarjetas >= umbral:
                    corrected = f"Más de {v}.5 tarjetas"
                    autocorrections.append(("tarjeta_menos_mas", v, corrected))
                    avisos_autocorregidos.append(
                        f"⚠️ AUTO-CORRECCIÓN APLICADA: 'Menos de {umbral:.1f} tarjetas' → '{corrected}' "
                        f"(árbitro promedia {arb_tarjetas:.1f}/partido)."
                    )
            except ValueError:
                pass
        for v in tarj_mas_vals:
            try:
                umbral = float(v) + 0.5
                if arb_tarjetas < umbral - 1.0:  # Only flag if significantly off
                    corrected = f"Menos de {v}.5 tarjetas"
                    autocorrections.append(("tarjeta_mas_menos", v, corrected))
                    avisos_autocorregidos.append(
                        f"⚠️ AUTO-CORRECCIÓN APLICADA: 'Más de {umbral:.1f} tarjetas' → '{corrected}' "
                        f"(árbitro promedia solo {arb_tarjetas:.1f}/partido)."
                    )
            except ValueError:
                pass

    # Validate and auto-correct corners markets
    corners_ref = None
    try:
        if comp_ctx is not None:
            cv = patron.get("corners_esperados_total")
            if cv is not None:
                corners_base = float(cv)
                corners_adj = comp_ctx.get('corners_ajuste', 0)
                corners_ref = corners_base + corners_adj
    except (TypeError, ValueError, AttributeError):
        pass

    if corners_ref is not None:
        corners_menos_vals = re.findall(r'[Mm]enos de (\d+(?:\.\d+)?)[\.,]?5? corner', picks_text)
        corners_mas_vals   = re.findall(r'[Mm][áa]s de (\d+(?:\.\d+)?)[\.,]?5? corner', picks_text)
        for v in corners_menos_vals:
            try:
                umbral = float(v) + 0.5
                if corners_ref >= umbral:
                    corrected = f"Más de {v}.5 corners"
                    autocorrections.append(("corner_menos_mas", v, corrected))
                    avisos_autocorregidos.append(
                        f"⚠️ AUTO-CORRECCIÓN APLICADA: 'Menos de {umbral:.1f} corners' → '{corrected}' "
                        f"(se esperan {corners_ref:.1f} corners)."
                    )
            except ValueError:
                pass
        for v in corners_mas_vals:
            try:
                umbral = float(v) + 0.5
                if corners_ref < umbral - 1.0:  # Only flag if significantly off
                    corrected = f"Menos de {v}.5 corners"
                    autocorrections.append(("corner_mas_menos", v, corrected))
                    avisos_autocorregidos.append(
                        f"⚠️ AUTO-CORRECCIÓN APLICADA: 'Más de {umbral:.1f} corners' → '{corrected}' "
                        f"(se esperan solo {corners_ref:.1f} corners)."
                    )
            except ValueError:
                pass

    # Validate and auto-correct goles markets (existing logic)
    goles_ref = h2h_goles_api
    if goles_ref is None:
        try:
            cv = patron.get("goles_esperados_total")
            if cv is not None:
                goles_ref = float(cv)
        except (TypeError, ValueError):
            pass

    if goles_ref is not None:
        for v in goles_menos_vals:
            try:
                umbral = float(v) + 0.5
                if goles_ref >= umbral + 0.5:
                    corrected = f"Más de {v}.5 goles"
                    autocorrections.append(("gol_menos_mas", v, corrected))
                    avisos_autocorregidos.append(
                        f"⚠️ AUTO-CORRECCIÓN APLICADA: 'Menos de {umbral:.1f} goles' → '{corrected}' "
                        f"(media = {goles_ref:.1f})."
                    )
            except ValueError:
                pass
        for v in goles_mas_vals:
            try:
                umbral = float(v) + 0.5
                if goles_ref < umbral - 1.0:
                    patron_buscar   = re.compile(
                        rf'[Mm][áa]s de {re.escape(v)}[\.,]?5? goles?', re.IGNORECASE
                    )
                    texto_mega_nuevo = patron_buscar.sub(f"Menos de {v}.5 goles", texto_mega)
                    if texto_mega_nuevo != texto_mega:
                        texto_mega = texto_mega_nuevo
                        logger.warning(f"Auto-corrección megacombinada: H2H={goles_ref:.1f} < umbral {umbral:.1f}")
                        autocorrections.append(("gol_mas_menos_auto", v, f"Menos de {v}.5 goles"))
                        avisos_autocorregidos.append(
                            f"⚠️ AUTO-CORRECCIÓN APLICADA: 'Más de {umbral:.1f} goles' → 'Menos de {umbral:.1f} goles' "
                            f"(media = {goles_ref:.1f})."
                        )
                    else:
                        incompatibilidades.append(
                            f"❌ INCOHERENTE: 'Más de {umbral:.1f} goles' pero la media es solo {goles_ref:.1f}."
                        )
            except ValueError:
                pass

    # Apply auto-corrections if any were found
    if autocorrections:
        # Apply corrections in reverse order to avoid index issues
        for correction_type, original_val, corrected_text in reversed(autocorrections):
            if correction_type.endswith("_auto"):
                # Already applied in the goles section above
                continue
            elif "tarjeta" in correction_type:
                if "menos_mas" in correction_type:
                    pattern = rf'[Mm]enos de {re.escape(original_val)}[\.,]?5? tarjeta'
                else:  # mas_menos
                    pattern = rf'[Mm][áa]s de {re.escape(original_val)}[\.,]?5? tarjeta'
            elif "corner" in correction_type:
                if "menos_mas" in correction_type:
                    pattern = rf'[Mm]enos de {re.escape(original_val)}[\.,]?5? corner'
                else:  # mas_menos
                    pattern = rf'[Mm][áa]s de {re.escape(original_val)}[\.,]?5? corner'
            else:  # gol
                if "menos_mas" in correction_type:
                    pattern = rf'[Mm]enos de {re.escape(original_val)}[\.,]?5? goles?'
                else:  # mas_menos
                    pattern = rf'[Mm][áa]s de {re.escape(original_val)}[\.,]?5? goles?'

            texto_mega = re.sub(pattern, corrected_text, texto_mega, flags=re.IGNORECASE)

    if not incompatibilidades and not avisos_autocorregidos:
        return texto_mega

    bloque = ""
    if avisos_autocorregidos:
        bloque += (
            "\n\n✅ *CORRECCIONES AUTOMÁTICAS APLICADAS:*\n"
            + "\n".join(f"  {a}" for a in avisos_autocorregidos)
        )
    if incompatibilidades:
        bloque += (
            "\n\n⚠️ *ATENCIÓN — PICKS INCOMPATIBLES DETECTADOS:*\n"
            + "\n".join(f"  {i}" for i in incompatibilidades)
            + "\n\n🚫 *No apuestes esta combinada tal como está. Revisa y elimina los picks marcados con ❌.*"
        )
    return texto_mega + bloque


def format_matches_keyboard(matches: list, page: int = 0) -> InlineKeyboardMarkup:
    total   = len(matches)
    pages   = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page    = max(0, min(page, pages - 1))
    start   = page * PAGE_SIZE
    end     = min(start + PAGE_SIZE, total)
    chunk   = matches[start:end]

    keyboard = []
    for m in chunk:
        icon  = "🔴" if m.get("is_live") else "⚽"
        score = f" {m.get('home_goals',0)}-{m.get('away_goals',0)}" if m.get("is_live") else ""
        label = f"{icon} {m['time']}{score} | {m['home'][:12]} vs {m['away'][:12]}"
        cb    = f"analyze|{m['fixture_id']}|{m['home'][:15]}|{m['away'][:15]}"
        keyboard.append([InlineKeyboardButton(label, callback_data=cb)])

    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton(f"◀ Ant ({page})", callback_data=f"page|{page-1}"))
    nav.append(InlineKeyboardButton(f"🔄 {start+1}-{end}/{total}", callback_data="refresh_matches"))
    if page < pages - 1:
        nav.append(InlineKeyboardButton(f"Sig ({page+2}) ▶", callback_data=f"page|{page+1}"))
    keyboard.append(nav)

    return InlineKeyboardMarkup(keyboard)

# ═══════════════════════════════════════════════════════════
#  JOBS: MOTOR AUTÓNOMO DE 2 FASES
# ═══════════════════════════════════════════════════════════
async def job_daily_scanner(context: ContextTypes.DEFAULT_TYPE) -> None:
    global _scanner_queue, _scanner_running, _scanner_total, _scanner_analyzed, _scanner_found
    if _scanner_running: return

    matches = get_todays_matches()
    candidatos = [m for m in matches if m["status"] in ACTIVE_STATUSES and m["mins_to_kickoff"] > 0]

    VIP_CODES = set(ESPN_LEAGUES.keys())
    VIP_IDS   = set(ALL_LEAGUES.keys())

    def _priority(m):
        is_vip = (m.get("league_id", 0) in VIP_IDS) or (m.get("espn_league_code", "") in VIP_CODES)
        return (0 if is_vip else 1, m.get("mins_to_kickoff", 9999))

    candidatos = sorted(candidatos, key=_priority)
    _scanner_queue = candidatos[:SCANNER_MAX_MATCHES]

    logger.info(f"Escáner: {len(candidatos)} candidatos totales, {len(_scanner_queue)} en cola (VIP primero)")

    if not _scanner_queue: return

    _scanner_total = len(_scanner_queue)
    _scanner_analyzed = 0
    _scanner_found = 0
    _scanner_running = True
    TARGET_MATCHES.clear()

    await notify_users(context, f"🔍 *ESCÁNER FASE 1 INICIADO*\nAnalizando {_scanner_total} partidos de hoy. Recibirás el resumen al terminar.")
    context.job_queue.run_once(_scanner_step, when=1, name="scanner_step")

async def _scanner_step(context: ContextTypes.DEFAULT_TYPE) -> None:
    global _scanner_queue, _scanner_running, _scanner_analyzed, _scanner_found

    if not _scanner_queue:
        _scanner_running = False
        if TARGET_MATCHES:
            lista = "\n".join(
                f"  ⚽ {d['match']['home']} vs {d['match']['away']} ({d['match']['league']}) — {get_minutes_to_kickoff(d['match']['date_iso'])} min"
                for d in TARGET_MATCHES.values()
            )
            resumen = (
                "✅ *Fase 1 Completada*\n"
                f"📊 Revisados: *{_scanner_analyzed}/{_scanner_total}*\n"
                f"🎯 En Radar: *{len(TARGET_MATCHES)}*\n\n"
                f"{lista}\n\n"
                "_Usa /radar para ver detalles._"
            )
        else:
            resumen = (
                "✅ *Fase 1 Completada*\n"
                f"📊 Revisados: *{_scanner_analyzed}/{_scanner_total}*\n"
                "📭 Ningún partido entró en el radar hoy."
            )
        await notify_users(context, resumen)
        return

    match = _scanner_queue.pop(0)
    _scanner_analyzed += 1
    resultado = await analyze_phase_1(match)

    if resultado == "ESTELAR":
        _scanner_found += 1
        TARGET_MATCHES[match["fixture_id"]] = {"match": match, "deep_analyzed": False}

    context.job_queue.run_once(_scanner_step, when=SCANNER_INTERVAL, name="scanner_step")

async def job_phase2_45_mins_check(context: ContextTypes.DEFAULT_TYPE) -> None:
    if not TARGET_MATCHES: return

    now_mono = time.monotonic()
    if now_mono < _groq_paused_until:
        logger.info(f"Fase 2 pospuesta — Groq en pausa {int(_groq_paused_until - now_mono)}s")
        return

    for fixture_id, data in list(TARGET_MATCHES.items()):
        if data["deep_analyzed"]: continue

        match = data["match"]
        mins_left = get_minutes_to_kickoff(match["date_iso"])

        if 0 < mins_left <= 55:
            TARGET_MATCHES[fixture_id]["deep_analyzed"] = True
            logger.info(f"Fase 2: {match['home']} vs {match['away']} ({mins_left} min)")
            deep_data = get_deep_data(fixture_id, match["league_id"])
            analisis = await analyze_phase_2_trader(match, deep_data, is_manual=False)

            if analisis and "❌" not in analisis:
                await notify_users(context, analisis)
            else:
                TARGET_MATCHES[fixture_id]["deep_analyzed"] = False
                logger.warning(f"Fase 2 fallida para {match['home']} vs {match['away']} — reintentará")
            break

async def job_update_results(context: ContextTypes.DEFAULT_TYPE) -> None:
    """V11 §2.5 — cierra el ciclo de backtesting: busca el resultado real de
    los fixtures con predicción registrada y lo guarda en `results`.
    Con eso /calibracion puede calcular Brier score y % de acierto."""
    pendientes = db_pending_results(max_rows=20)
    if not pendientes:
        return
    logger.info(f"Backtesting: buscando resultado de {len(pendientes)} fixtures...")
    loop = asyncio.get_running_loop()
    for fixture_id in pendientes:
        try:
            data = await loop.run_in_executor(
                None, lambda fid=fixture_id: football_api("fixtures", {"id": fid}, ttl=1800))
            if not data or not data.get("response"):
                continue
            f = data["response"][0]
            if f["fixture"]["status"]["short"] not in FINISHED_STATUSES:
                continue
            gl = f["goals"]["home"]
            gv = f["goals"]["away"]
            if gl is None or gv is None:
                continue
            db_save_result(fixture_id, int(gl), int(gv))
            logger.info(f"Backtesting: resultado {fixture_id} = {gl}-{gv} guardado")
        except Exception as e:
            logger.error(f"job_update_results fixture {fixture_id}: {e}")

# ═══════════════════════════════════════════════════════════
#  COMANDOS
# ═══════════════════════════════════════════════════════════
async def calibracion_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Brier score y acierto por mercado — la diferencia entre 'suena
    convincente' y 'medible' (V11 §2.5)."""
    cal = db_calibration()
    with _db_lock, _db() as c:
        n_pred = c.execute("SELECT COUNT(*) AS n FROM predictions").fetchone()["n"]
        n_res  = c.execute("SELECT COUNT(*) AS n FROM results").fetchone()["n"]
    if not cal:
        await update.message.reply_text(
            f"📈 *CALIBRACIÓN*\n\n"
            f"Predicciones registradas: {n_pred}\n"
            f"Resultados recogidos: {n_res}\n\n"
            "Aún no hay pares predicción+resultado suficientes. "
            "El bot registra cada señal automáticamente; vuelve tras unas jornadas.",
            parse_mode="Markdown")
        return
    lineas = []
    NOMBRES = {"1X2": "1X2 (ganador)", "over_2_5": "Más de 2.5 goles", "btts": "BTTS"}
    for market, m in sorted(cal.items()):
        lineas.append(
            f"• *{NOMBRES.get(market, market)}* — n={m['n']}\n"
            f"   Acierto: {m['hit_rate']:.0%} | Brier: {m['brier']:.3f} "
            f"{'🟢' if m['brier'] < 0.20 else '🟡' if m['brier'] < 0.25 else '🔴'}"
        )
    await update.message.reply_text(
        "📈 *CALIBRACIÓN DEL MODELO*\n"
        "_Brier: 0 = perfecto, 0.25 = azar en mercado binario_\n\n"
        + "\n".join(lineas)
        + f"\n\n📦 Total: {n_pred} predicciones, {n_res} resultados",
        parse_mode="Markdown")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ensure_user_active(update.effective_chat.id)
    msg = (
        "🤖 *ScoutBet V10.0 — Multi-Ronda + Contexto Competicional*\n\n"
        "🔄 Motor multi-ronda: 3 consultas Groq antes del análisis\n"
        "   R1 → Patrón del partido (goles, corners, tarjetas)\n"
        "   R2 → Scout jugadores (goleadores, porteros, tiradores)\n"
        "   R3 → Síntesis predictiva + señal de valor\n"
        "✅ Contexto competicional: ajuste automático por fase/copa\n"
        "✅ Todos los mercados: tiros, paradas portero, corners por equipo,\n"
        "   tarjetas por jugador, anytime scorer, marcador exacto y más\n\n"
        "📋 *COMANDOS:*\n"
        "/escanear — Lanza Fase 1 ahora\n"
        "/escaner\\_estado — Progreso del escáner\n"
        "/radar — Partidos cazados para Fase 2\n"
        "/partidos — Partidos próximas 24 h (incluye en vivo)\n"
        "/vivo — Partidos en juego ahora\n"
        "/debug — Estado de APIs y Groq\n"
        "/frecuencia <min> /alerta /parar — Control alertas\n\n"
        "_Envía 'Equipo A vs Equipo B' para análisis manual_"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def frecuencia_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = str(update.effective_chat.id)
    ensure_user_active(update.effective_chat.id)
    try:
        minutos = int(context.args[0])
        if not (1 <= minutos <= 1440):
            await update.message.reply_text("❌ Valor entre 1 y 1440 minutos.")
            return
        subscribed_users[chat_id]["interval"] = minutos
        save_users(subscribed_users)
        await update.message.reply_text(f"✅ Frecuencia ajustada a *{minutos} min*.", parse_mode="Markdown")
    except:
        await update.message.reply_text("Usa: `/frecuencia 15`", parse_mode="Markdown")

async def alerta_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ensure_user_active(update.effective_chat.id)
    await update.message.reply_text("✅ *Alertas ACTIVADAS*.", parse_mode="Markdown")

async def parar_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = str(update.effective_chat.id)
    if chat_id in subscribed_users:
        subscribed_users[chat_id]["active"] = False
        save_users(subscribed_users)
    await update.message.reply_text("🔕 Alertas DESACTIVADAS.")

async def debug_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    status_api = "🔴 AGOTADA (ESPN activo)" if API_FOOTBALL_EXHAUSTED else "🟢 OK"
    now_mono = time.monotonic()

    keys_lines = []
    for i in range(len(GROQ_API_KEYS)):
        estado = _breaker_state(i)
        if estado == "OPEN_QUOTA":
            st = "💀 OPEN (cuota agotada)"
        elif estado == "OPEN":
            st = f"🔴 OPEN {int(_key_paused_until[i] - now_mono)}s"
        elif estado == "HALF_OPEN":
            st = "🟡 HALF-OPEN (probando)"
        else:
            st = "🟢 CLOSED"
        keys_lines.append(f"  Key#{i+1}: {st} (aperturas: {_key_opens_count[i]})")
    groq_keys_txt = "\n".join(keys_lines)

    if _groq_inflight > 0:
        groq_global = f"🟡 Procesando ({_groq_inflight} en curso)..."
    elif all(_key_quota_dead):
        groq_global = "💀 Todas las keys agotadas"
    elif all(now_mono < _key_paused_until[i] for i in range(len(GROQ_API_KEYS))):
        groq_global = "🔴 Todas en pausa"
    else:
        groq_global = "🟢 Rotador activo"

    sofa_ok = bool(_get("https://www.sofascore.com/api/v1/sport/football/scheduled-events/" + now_spain().strftime("%Y-%m-%d"), is_json=True))
    espn_ok = bool(_get("https://site.api.espn.com/apis/site/v2/sports/soccer/esp.1/scoreboard", is_json=True))

    txt = (
        f"🛠 *DIAGNÓSTICO V11.0*\n\n"
        f"📡 API-Football: {status_api}\n"
        f"📡 ESPN: {'🟢 OK' if espn_ok else '🔴 Error'}\n"
        f"📡 SofaScore: {'🟢 OK' if sofa_ok else '🔴 Error'}\n"
        f"🧠 Groq ({GROQ_MODEL}): {groq_global}\n"
        f"{groq_keys_txt}\n"
        f"🔄 Motor: Multi-ronda (R1 patrón + R2 scout + R3 síntesis)\n"
        f"⚙️ Escáner: {'En marcha' if _scanner_running else 'Detenido'} ({_scanner_analyzed}/{_scanner_total})\n"
        f"🎯 Partidos en Radar: {len(TARGET_MATCHES)}\n\n"
        f"📊 *Métricas sesión:*\n"
        f"  Groq ok/err: {METRICS['groq_calls_ok']}/{METRICS['groq_calls_error']} | "
        f"429: {METRICS['groq_rate_limited']} | breaker: {METRICS['breaker_opened']}\n"
        f"  JSON inválido: {METRICS['groq_json_invalid']} | esquema rechazado: {METRICS['groq_schema_rejected']}\n"
        f"  CoVe correcciones: {METRICS['cove_correcciones']} | SC↓confianza: {METRICS['selfconsistency_baja_confianza']}\n"
        f"  Fallback ESPN: {METRICS['espn_fallback_used']} | Preds/Results: "
        f"{METRICS['predicciones_registradas']}/{METRICS['resultados_registrados']}"
    )
    await update.message.reply_text(txt, parse_mode="Markdown")

async def resetapi_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Resetea el flag API_FOOTBALL_EXHAUSTED y limpia caché de lineups para forzar reconsulta."""
    global API_FOOTBALL_EXHAUSTED
    API_FOOTBALL_EXHAUSTED = False
    # Limpiar entradas de caché de lineups para forzar reconsulta fresca
    lineup_keys = [k for k in list(_api_cache.keys()) if k.startswith("fixtures/lineups:")]
    for k in lineup_keys:
        _api_cache.pop(k, None)
        _api_cache_ts.pop(k, None)
    await update.message.reply_text(
        f"✅ *API-Football reseteada*\n"
        f"🗑 Caché de alineaciones limpiada ({len(lineup_keys)} entradas borradas)\n"
        f"Vuelve a pedir el análisis del partido para obtener las alineaciones frescas.",
        parse_mode="Markdown"
    )

async def partidos_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_chat_action(ChatAction.TYPING)
    matches = get_todays_matches()
    if not matches:
        await update.message.reply_text("📭 No se encontraron partidos.")
        return
    context.user_data["cached_matches"] = matches
    context.user_data["cached_matches_ts"] = time.monotonic()
    total = len(matches)
    pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    await update.message.reply_text(
        f"📅 *Próximas 24 h — {total} partidos ({pages} páginas):*",
        parse_mode="Markdown",
        reply_markup=format_matches_keyboard(matches, page=0)
    )

async def vivo_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_chat_action(ChatAction.TYPING)
    live = get_live_matches()
    if not live:
        await update.message.reply_text("🔴 No hay partidos en juego.")
        return
    await update.message.reply_text("🔴 *En vivo:*", parse_mode="Markdown", reply_markup=format_matches_keyboard(live))

async def escanear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ensure_user_active(update.effective_chat.id)
    if _scanner_running:
        await update.message.reply_text(f"⏳ Fase 1 en marcha ({_scanner_analyzed}/{_scanner_total}).")
        return
    await job_daily_scanner(context)

async def escaner_estado_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _scanner_running and _scanner_total == 0:
        await update.message.reply_text("📭 Escáner detenido.")
        return
    await update.message.reply_text(
        f"📊 *Progreso Fase 1:*\nAnalizados: {_scanner_analyzed}/{_scanner_total}\nCazados: {_scanner_found}",
        parse_mode="Markdown"
    )

async def radar_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pendientes = [d["match"] for d in TARGET_MATCHES.values() if not d["deep_analyzed"]]
    if not pendientes:
        await update.message.reply_text("📭 Radar vacío.")
        return
    txt = f"🎯 *RADAR ({len(pendientes)} partidos esperando Fase 2):*\n\n"
    for m in pendientes:
        txt += f"⚽ {m['home']} vs {m['away']} (Faltan {get_minutes_to_kickoff(m['date_iso'])} min)\n"
    await update.message.reply_text(txt, parse_mode="Markdown")

# ═══════════════════════════════════════════════════════════
#  HANDLERS: BOTONES Y TEXTO
# ═══════════════════════════════════════════════════════════
async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    if query.data.startswith("page|"):
        page = int(query.data.split("|")[1])
        cached   = context.user_data.get("cached_matches")
        cache_ts = context.user_data.get("cached_matches_ts", 0)
        if cached and time.monotonic() - cache_ts < 300:
            matches = cached
        else:
            matches = get_todays_matches()
            context.user_data["cached_matches"] = matches
            context.user_data["cached_matches_ts"] = time.monotonic()
        await query.edit_message_reply_markup(
            reply_markup=format_matches_keyboard(matches, page=page)
        )
        return

    if query.data == "refresh_matches":
        matches = get_todays_matches()
        context.user_data["cached_matches"] = matches
        context.user_data["cached_matches_ts"] = time.monotonic()
        total = len(matches)
        pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
        await query.edit_message_text(
            f"📅 *Próximas 24 h — {total} partidos ({pages} páginas):*",
            parse_mode="Markdown",
            reply_markup=format_matches_keyboard(matches, page=0)
        )
        return

    if query.data.startswith("analyze|"):
        if time.monotonic() < _groq_paused_until:
            await query.edit_message_text(f"⚠️ Groq enfriando. Faltan {int(_groq_paused_until - time.monotonic())}s.")
            return

        parts = query.data.split("|", 3)
        fixture_id = int(parts[1])
        home, away = parts[2], parts[3]

        await query.edit_message_text(
            f"⏳ Analizando *{home} vs {away}* (V10.0 multi-ronda)...\n"
            f"_(R1: patrón → R2: scout jugadores → R3: síntesis)_",
            parse_mode="Markdown"
        )

        match_data = next((m for m in get_todays_matches() if m["fixture_id"] == fixture_id), None)
        if not match_data: match_data = next((m for m in get_live_matches() if m["fixture_id"] == fixture_id), None)
        if not match_data:
            match_data = {
                "fixture_id": fixture_id, "home": home, "away": away,
                "is_live": False, "league_id": 0, "league": "Manual",
                "venue": None, "referee": None, "home_goals": 0, "away_goals": 0, "minute": None
            }

        deep_data = get_deep_data(fixture_id, match_data["league_id"])
        result = await analyze_phase_2_trader(match_data, deep_data, is_manual=True)

        for chunk in split_message(result):
            sent = False
            for attempt in range(3):
                try:
                    await asyncio.wait_for(
                        context.bot.send_message(chat_id=query.from_user.id, text=chunk, parse_mode="Markdown"),
                        timeout=20
                    )
                    sent = True
                    break
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout enviando chunk Markdown (intento {attempt+1}/3)")
                    await asyncio.sleep(2)
                except Exception as e:
                    logger.warning(f"Error enviando chunk con Markdown: {e} — cambiando a texto plano")
                    break
            if not sent:
                clean_chunk = chunk.replace("*", "").replace("_", "").replace("`", "").replace("[", "").replace("]", "")
                for attempt in range(3):
                    try:
                        await asyncio.wait_for(
                            context.bot.send_message(chat_id=query.from_user.id, text=clean_chunk),
                            timeout=20
                        )
                        sent = True
                        break
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout enviando chunk texto plano (intento {attempt+1}/3)")
                        await asyncio.sleep(2)
                    except Exception as e2:
                        logger.error(f"Error definitivo enviando chunk: {e2}")
                        break
            if not sent:
                logger.error("No se pudo enviar un chunk tras 6 intentos — se omite.")

async def text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text
    if " vs " not in text.lower(): return

    if time.monotonic() < _groq_paused_until:
        await update.message.reply_text(f"⚠️ Groq enfriando. Espera {int(_groq_paused_until - time.monotonic())}s.")
        return

    parts = re.split(r"(?i)\s+vs\s+", text)
    if len(parts) != 2:
        await update.message.reply_text("❌ Formato: EquipoA vs EquipoB")
        return

    team_a, team_b = parts[0].strip(), parts[1].strip()
    status = await update.message.reply_text(
        f"🔍 Analizando *{team_a} vs {team_b}*...\n"
        f"_(R1: patrón del partido → R2: scout jugadores → R3: síntesis → análisis final)_",
        parse_mode="Markdown"
    )

    match_data = next(
        (m for m in get_todays_matches()
         if _teams_match(team_a, m["home"]) or _teams_match(team_a, m["away"])),
        None
    )
    if not match_data:
        match_data = {
            "fixture_id": 0, "home": team_a.title(), "away": team_b.title(),
            "is_live": False, "league_id": 0, "league": "Manual",
            "venue": None, "referee": None, "home_goals": 0, "away_goals": 0, "minute": None
        }

    deep_data = get_deep_data(match_data["fixture_id"], match_data["league_id"])
    result = await analyze_phase_2_trader(match_data, deep_data, is_manual=True)

    await status.delete()
    for chunk in split_message(result):
        sent = False
        for attempt in range(3):
            try:
                await asyncio.wait_for(
                    update.message.reply_text(chunk, parse_mode="Markdown"),
                    timeout=20
                )
                sent = True
                break
            except asyncio.TimeoutError:
                logger.warning(f"Timeout enviando chunk Markdown texto (intento {attempt+1}/3)")
                await asyncio.sleep(2)
            except Exception as e:
                logger.warning(f"Error Markdown en text_message: {e} — cambiando a texto plano")
                break
        if not sent:
            clean_chunk = chunk.replace("*", "").replace("_", "").replace("`", "").replace("[", "").replace("]", "")
            for attempt in range(3):
                try:
                    await asyncio.wait_for(
                        update.message.reply_text(clean_chunk),
                        timeout=20
                    )
                    sent = True
                    break
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout enviando chunk texto plano (intento {attempt+1}/3)")
                    await asyncio.sleep(2)
                except Exception as e2:
                    logger.error(f"Error definitivo en text_message chunk: {e2}")
                    break

# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════
def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("frecuencia", frecuencia_command))
    app.add_handler(CommandHandler("alerta", alerta_command))
    app.add_handler(CommandHandler("parar", parar_command))
    app.add_handler(CommandHandler("partidos", partidos_command))
    app.add_handler(CommandHandler("vivo", vivo_command))
    app.add_handler(CommandHandler("debug", debug_command))
    app.add_handler(CommandHandler("resetapi", resetapi_command))
    app.add_handler(CommandHandler("escanear", escanear_command))
    app.add_handler(CommandHandler("escaner_estado", escaner_estado_command))
    app.add_handler(CommandHandler("radar", radar_command))
    app.add_handler(CommandHandler("calibracion", calibracion_command))

    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message))

    jq: JobQueue = app.job_queue
    utc_h_scan  = utc_for_spain_hour(9)
    utc_h_scan2 = utc_for_spain_hour(17)
    jq.run_daily(job_daily_scanner, time=dt_time(hour=utc_h_scan,  minute=0), name="daily_scanner")
    jq.run_daily(job_daily_scanner, time=dt_time(hour=utc_h_scan2, minute=0), name="daily_scanner_evening")
    jq.run_repeating(job_phase2_45_mins_check, interval=300, first=30, name="phase2_check")
    jq.run_repeating(job_update_results, interval=6 * 3600, first=120, name="update_results")

    utc_h_reset = utc_for_spain_hour(2)
    jq.run_daily(
        lambda c: (globals().update(API_FOOTBALL_EXHAUSTED=False), clear_daily_notifications()),
        time=dt_time(hour=utc_h_reset, minute=0),
        name="reset"
    )

    async def _set_commands(app_ref):
        from telegram import BotCommand
        await app_ref.bot.set_my_commands([
            BotCommand("partidos",       "⚽ Partidos próximas 24 h"),
            BotCommand("vivo",           "🔴 Partidos en juego ahora"),
            BotCommand("escanear",       "🔍 Lanzar escáner Fase 1"),
            BotCommand("escaner_estado", "📊 Progreso del escáner"),
            BotCommand("radar",          "🎯 Partidos en el radar"),
            BotCommand("alerta",         "🔔 Activar alertas automáticas"),
            BotCommand("parar",          "🔕 Desactivar alertas"),
            BotCommand("frecuencia",     "⏱ Cambiar frecuencia de alertas"),
            BotCommand("calibracion",    "📈 Brier score y acierto por mercado"),
            BotCommand("debug",          "🛠 Estado de APIs y Groq"),
            BotCommand("start",          "🤖 Bienvenida y ayuda"),
        ])

    app.post_init = _set_commands

    logger.info(f"ScoutBet V11.0 (Híbrido Cuant+IA · SQLite · Breaker/key · Self-consistency x{SELF_CONSISTENCY_N} · CoVe · Backtesting) iniciado — {len(GROQ_API_KEYS)} keys Groq.")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()
