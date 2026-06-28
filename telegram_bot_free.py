"""
telegram_bot_free.py - versión fácil con botones y búsqueda flexible

Sustituye al telegram_bot_free.py anterior.

Mejoras:
- No hace falta escribir formato largo.
- /proximos muestra botones de partidos.
- /hoy muestra partidos de hoy.
- Puedes escribir: brasil japon
- Puedes escribir: south africa canada
- Tolera tildes, mayúsculas/minúsculas y algunos errores.
- Si no encuentra exacto, propone partidos parecidos.
- Sigue aceptando CSV.
"""

from __future__ import annotations

import os
import sys
import html
import traceback
import unicodedata
from pathlib import Path
from datetime import datetime, date

import pandas as pd

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from predict import predict_csv as predict_batch
from config import PATHS


JOBS_DIR = PROJECT_DIR / "telegram_jobs"
JOBS_DIR.mkdir(exist_ok=True)

FIXTURES_FILE = PROJECT_DIR / "telegram_fixtures_worldcup.csv"

REQUIRED_COLUMNS = [
    "equipo_local",
    "equipo_visitante",
    "sede",
    "fase",
    "fecha",
    "dias_descanso_local",
    "dias_descanso_visitante",
]


# ---------------------------------------------------------
# Seguridad
# ---------------------------------------------------------

def _allowed_user(user_id: int) -> bool:
    allowed = os.getenv("ALLOWED_TELEGRAM_USERS", "").strip()
    if not allowed:
        return True
    ids = {x.strip() for x in allowed.split(",") if x.strip()}
    return str(user_id) in ids


async def _guard(update: Update) -> bool:
    user = update.effective_user
    msg = update.effective_message
    if user and not _allowed_user(user.id):
        if msg:
            await msg.reply_text("⛔ No estás autorizado para usar este bot.")
        return False
    return True


# ---------------------------------------------------------
# Utilidades texto / matching
# ---------------------------------------------------------

def _strip_accents(text: str) -> str:
    text = str(text or "")
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def _norm(text: str) -> str:
    text = _strip_accents(str(text or "").lower())
    replacements = {
        "-": " ",
        "_": " ",
        "|": " ",
        "/": " ",
        "\\": " ",
        ".": " ",
        ",": " ",
        ":": " ",
        ";": " ",
        "  ": " ",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


ALIASES = {
    "brasil": "brazil",
    "japon": "japan",
    "japón": "japan",
    "alemania": "germany",
    "paraguay": "paraguay",
    "sudafrica": "south africa",
    "sudáfrica": "south africa",
    "southafrica": "south africa",
    "canada": "canada",
    "canadá": "canada",
    "francia": "france",
    "suecia": "sweden",
    "espana": "spain",
    "españa": "spain",
    "austria": "austria",
    "holanda": "netherlands",
    "paises bajos": "netherlands",
    "países bajos": "netherlands",
    "netherlands": "netherlands",
    "marruecos": "morocco",
    "moroco": "morocco",
    "marrocos": "morocco",
    "costa de marfil": "ivory coast",
    "ivory coast": "ivory coast",
    "noruega": "norway",
    "argentina": "argentina",
    "cabo verde": "cape verde",
    "cape verde": "cape verde",
    "belgica": "belgium",
    "bélgica": "belgium",
    "senegal": "senegal",
    "inglaterra": "england",
    "dr congo": "dr congo",
    "congo": "dr congo",
    "eeuu": "united states",
    "usa": "united states",
    "estados unidos": "united states",
    "bosnia": "bosnia and herzegovina",
    "portugal": "portugal",
    "croacia": "croatia",
    "suiza": "switzerland",
    "argelia": "algeria",
    "australia": "australia",
    "egipto": "egypt",
    "colombia": "colombia",
    "ghana": "ghana",
    "gana": "ghana",
    "mexico": "mexico",
    "méxico": "mexico",
    "ecuador": "ecuador",
}


def _alias_text(text: str) -> str:
    text_norm = _norm(text)
    # Reemplazo de frases largas primero
    for k in sorted(ALIASES.keys(), key=len, reverse=True):
        kn = _norm(k)
        if kn in text_norm:
            text_norm = text_norm.replace(kn, ALIASES[k])
    return _norm(text_norm)


def _safe_text(value) -> str:
    if value is None:
        return ""
    return html.escape(str(value))


def _pct(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "N/A"


def _check_local_system() -> tuple[bool, list[str]]:
    problems = []

    if not PATHS["dataset"].exists():
        problems.append(f"Falta dataset ({PATHS['dataset'].name}). Ejecuta: python build_dataset.py --no-statsbomb")

    if not PATHS["team_state"].exists():
        problems.append(f"Falta team_state ({PATHS['team_state'].name}). Ejecuta: python build_dataset.py --no-statsbomb")

    if not PATHS["artifact"].exists():
        problems.append(f"Falta modelo ({PATHS['artifact'].name}). Ejecuta: python train.py")

    return len(problems) == 0, problems


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

def _default_fixtures() -> pd.DataFrame:
    rows = [
        ["53452545", "South Africa", "Canada", "Los Angeles", "Dieciseisavos", "2026-06-28", "21:00", 7, 7],
        ["53452557", "Brazil", "Japan", "Houston", "Dieciseisavos", "2026-06-29", "19:00", 7, 7],
        ["53452541", "Germany", "Paraguay", "Boston", "Dieciseisavos", "2026-06-29", "22:30", 7, 7],
        ["53452547", "Netherlands", "Morocco", "Monterrey", "Dieciseisavos", "2026-06-30", "03:00", 7, 7],
        ["53452561", "Ivory Coast", "Norway", "Dallas", "Dieciseisavos", "2026-06-30", "19:00", 7, 7],
        ["53452543", "France", "Sweden", "Nueva York", "Dieciseisavos", "2026-06-30", "23:00", 7, 7],
        ["53452563", "Mexico", "Ecuador", "Ciudad de Mexico", "Dieciseisavos", "2026-07-01", "03:00", 7, 7],
        ["53452565", "England", "DR Congo", "Atlanta", "Dieciseisavos", "2026-07-01", "18:00", 7, 7],
        ["53452555", "Belgium", "Senegal", "Seattle", "Dieciseisavos", "2026-07-01", "22:00", 7, 7],
        ["53452553", "United States", "Bosnia and Herzegovina", "San Francisco", "Dieciseisavos", "2026-07-02", "02:00", 7, 7],
        ["53452551", "Spain", "Austria", "Los Angeles", "Dieciseisavos", "2026-07-02", "21:00", 7, 7],
        ["53452549", "Portugal", "Croatia", "Toronto", "Dieciseisavos", "2026-07-03", "01:00", 7, 7],
        ["53452505", "Switzerland", "Algeria", "Vancouver", "Dieciseisavos", "2026-07-03", "05:00", 7, 7],
        ["53452503", "Australia", "Egypt", "Dallas", "Dieciseisavos", "2026-07-03", "20:00", 7, 7],
        ["53452569", "Argentina", "Cape Verde", "Miami", "Dieciseisavos", "2026-07-04", "00:00", 7, 7],
        ["53452507", "Colombia", "Ghana", "Kansas City", "Dieciseisavos", "2026-07-04", "03:30", 7, 7],
    ]
    return pd.DataFrame(rows, columns=[
        "game_id",
        "equipo_local",
        "equipo_visitante",
        "sede",
        "fase",
        "fecha",
        "hora_espana",
        "dias_descanso_local",
        "dias_descanso_visitante",
    ])


def load_fixtures() -> pd.DataFrame:
    if FIXTURES_FILE.exists():
        df = pd.read_csv(FIXTURES_FILE)
    else:
        df = _default_fixtures()
        df.to_csv(FIXTURES_FILE, index=False, encoding="utf-8")

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise RuntimeError(f"telegram_fixtures_worldcup.csv no tiene columna {col}")

    if "game_id" not in df.columns:
        df["game_id"] = [str(i) for i in range(len(df))]
    if "hora_espana" not in df.columns:
        df["hora_espana"] = ""

    return df.reset_index(drop=True)


def _fixture_label(row: pd.Series) -> str:
    hora = str(row.get("hora_espana", "") or "")
    fecha = str(row.get("fecha", "") or "")
    return f"{fecha} {hora} · {row['equipo_local']} vs {row['equipo_visitante']}"


def _fixture_to_row(row: pd.Series) -> dict:
    return {
        "equipo_local": row["equipo_local"],
        "equipo_visitante": row["equipo_visitante"],
        "sede": row["sede"],
        "fase": row["fase"],
        "fecha": row["fecha"],
        "dias_descanso_local": int(row.get("dias_descanso_local", 7) or 7),
        "dias_descanso_visitante": int(row.get("dias_descanso_visitante", 7) or 7),
    }


def find_fixture(text: str, limit: int = 5) -> pd.DataFrame:
    """
    Busca por texto libre.
    Ejemplos:
    - brasil japon
    - sudafrica canada
    - france sweden
    """
    df = load_fixtures()
    q = _alias_text(text)

    scores = []
    q_words = set(q.split())

    for i, row in df.iterrows():
        home = _alias_text(row["equipo_local"])
        away = _alias_text(row["equipo_visitante"])
        label = _alias_text(f"{row['equipo_local']} {row['equipo_visitante']} {row.get('fecha', '')} {row.get('sede', '')}")
        words = set(label.split())

        score = 0

        # Coincidencia fuerte por equipos
        if home in q and away in q:
            score += 100
        elif away in q and home in q:
            score += 100
        else:
            if home in q:
                score += 45
            if away in q:
                score += 45

        # Coincidencias por palabras
        score += len(q_words & words) * 8

        # Búsqueda parcial
        for w in q_words:
            if len(w) >= 4 and w in label:
                score += 4

        scores.append(score)

    out = df.copy()
    out["_score"] = scores
    out = out[out["_score"] > 0].sort_values("_score", ascending=False).head(limit)
    return out


# ---------------------------------------------------------
# Predicción
# ---------------------------------------------------------

def _run_prediction(rows: list[dict]) -> pd.DataFrame:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    input_path = JOBS_DIR / f"input_{ts}.csv"
    output_path = JOBS_DIR / f"output_{ts}.csv"

    df = pd.DataFrame(rows)
    df.to_csv(input_path, index=False, encoding="utf-8")

    return predict_batch(input_path, output_path)


def _norm_decision(value) -> str:
    """
    Normaliza la decisión que devuelve predict.py.
    Soporta tanto formato antiguo:
      OVER / UNDER / NO APOSTAR
    como formato nuevo:
      OVER 2.5 / UNDER 2.5 / NO APOSTAR
    """
    v = str(value or "").strip().upper()
    if v in {"OVER", "OVER 2.5", "APUESTA VALIDADA: OVER 2.5"}:
        return "OVER 2.5"
    if v in {"UNDER", "UNDER 2.5", "APUESTA VALIDADA: UNDER 2.5"}:
        return "UNDER 2.5"
    return "NO APOSTAR"


def _row_first(row: pd.Series, *names, default=None):
    """Devuelve el primer valor existente/no vacío de una fila."""
    for name in names:
        if name in row.index:
            value = row.get(name)
            if value is not None and not pd.isna(value):
                return value
    return default


def _format_prediction_row(row: pd.Series) -> str:
    # predict.py nuevo devuelve "decision"; versiones viejas podían devolver "prediccion".
    raw_pred = _row_first(row, "decision", "prediccion", default="NO APOSTAR")
    pred = _norm_decision(raw_pred)

    obs = _norm_decision(_row_first(row, "observacion_no_apuesta", default=""))
    partido = f"{row.get('equipo_local', '')} vs {row.get('equipo_visitante', '')}"

    bloqueos = str(row.get("bloqueos", "") or "").strip()
    checks = str(row.get("checks_ok", "") or "").strip()
    bloqueos_limpio = bloqueos.lower() in {"", "nan", "none", "ninguno"}

    if pred == "OVER 2.5":
        semaforo = "🟢"
        decision = "✅ APOSTAR OVER 2.5"
    elif pred == "UNDER 2.5":
        semaforo = "🟢"
        decision = "✅ APOSTAR UNDER 2.5"
    else:
        semaforo = "🟡"
        decision = "⛔ NO APOSTAR"

    text = []
    text.append(f"{semaforo} <b>{_safe_text(partido)}</b>")
    text.append(f"📅 Fecha: <b>{_safe_text(row.get('fecha', ''))}</b>")
    text.append(f"🏟️ Sede: <b>{_safe_text(row.get('sede', ''))}</b> | Fase: <b>{_safe_text(row.get('fase', ''))}</b>")
    text.append("")
    text.append(f"🎯 Decisión: <b>{_safe_text(decision)}</b>")
    text.append(f"👀 Observación bruta: <b>{_safe_text(obs)}</b>")
    text.append("")

    # Columnas nuevas del proyecto ultra.
    text.append(f"Poisson base Over: <b>{_pct(row.get('p_over_poisson_base'))}</b>")
    text.append(f"Poisson GAP Over: <b>{_pct(row.get('p_over_poisson_gap'))}</b>")
    text.append(f"ML Over: <b>{_pct(row.get('p_over_ml'))}</b>")
    text.append(f"Final Over: <b>{_pct(row.get('p_over_final'))}</b>")
    text.append(f"Final Under: <b>{_pct(row.get('p_under_final'))}</b>")

    conf_modelo = _row_first(row, "confianza_modelo", "confianza_final", default=None)
    conf_apuesta = _row_first(row, "confianza_apuesta", "confianza_final", default=None)
    text.append(f"Confianza modelo: <b>{_pct(conf_modelo)}</b>")
    text.append(f"Confianza apuesta: <b>{_pct(conf_apuesta)}</b>")

    if "filter_mode" in row.index:
        text.append(f"Modo filtro: <b>{_safe_text(row.get('filter_mode'))}</b>")

    text.append("")
    text.append(f"Elo: <b>{_safe_text(row.get('elo_local', ''))}</b> vs <b>{_safe_text(row.get('elo_visitante', ''))}</b>")
    text.append(f"Partidos previos: <b>{_safe_text(row.get('partidos_previos_local', ''))}</b> / <b>{_safe_text(row.get('partidos_previos_visitante', ''))}</b>")

    if checks and checks.lower() not in {"nan", "none"}:
        text.append("")
        text.append(f"✅ Checks: {_safe_text(checks)}")

    text.append("")
    if bloqueos_limpio:
        text.append("✅ Bloqueos: <b>ninguno</b>")
    else:
        text.append(f"⛔ Bloqueos: <b>{_safe_text(bloqueos)}</b>")

    return "\n".join(text)


async def analyze_fixture_by_index(update_or_query, context: ContextTypes.DEFAULT_TYPE, idx: int):
    ok, problems = _check_local_system()
    if not ok:
        msg = "⚠️ Primero arregla esto:\n" + "\n".join(f"• {_safe_text(p)}" for p in problems)
        if hasattr(update_or_query, "message") and update_or_query.message:
            await update_or_query.message.reply_html(msg)
        else:
            await update_or_query.edit_message_text(msg, parse_mode="HTML")
        return

    df = load_fixtures()
    if idx < 0 or idx >= len(df):
        await update_or_query.edit_message_text("No encuentro ese partido.")
        return

    row = df.iloc[idx]
    rows = [_fixture_to_row(row)]

    out = _run_prediction(rows)
    text = _format_prediction_row(out.iloc[0])

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🔙 Próximos", callback_data="menu:proximos"),
            InlineKeyboardButton("🏠 Inicio", callback_data="menu:inicio"),
        ]
    ])

    if hasattr(update_or_query, "edit_message_text"):
        await update_or_query.edit_message_text(text, parse_mode="HTML", reply_markup=keyboard)
    else:
        await update_or_query.message.reply_html(text, reply_markup=keyboard)


# ---------------------------------------------------------
# Teclados / menús
# ---------------------------------------------------------

def main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📅 Próximos partidos", callback_data="menu:proximos")],
        [InlineKeyboardButton("🔥 Brasil vs Japón", callback_data="match:1")],
        [InlineKeyboardButton("🇿🇦 South Africa vs Canada", callback_data="match:0")],
        [InlineKeyboardButton("ℹ️ Estado del sistema", callback_data="menu:estado")],
    ])


def fixtures_keyboard(df: pd.DataFrame, title: str = "proximos") -> InlineKeyboardMarkup:
    buttons = []
    for idx, row in df.iterrows():
        label = f"{row.get('fecha')} {row.get('hora_espana', '')} · {row['equipo_local']} vs {row['equipo_visitante']}"
        if len(label) > 60:
            label = label[:57] + "..."
        buttons.append([InlineKeyboardButton(label, callback_data=f"match:{idx}")])

    buttons.append([InlineKeyboardButton("🏠 Inicio", callback_data="menu:inicio")])
    return InlineKeyboardMarkup(buttons)


async def send_fixtures(update: Update, context: ContextTypes.DEFAULT_TYPE, mode: str = "proximos"):
    df = load_fixtures()

    if mode == "hoy":
        today = date.today().isoformat()
        sub = df[df["fecha"].astype(str) == today]
        if sub.empty:
            # Si la fecha del PC no coincide, muestra el primer día disponible.
            first_date = str(df["fecha"].min())
            sub = df[df["fecha"].astype(str) == first_date]
            title = f"📅 No hay partidos para la fecha del PC. Muestro {first_date}:"
        else:
            title = "📅 Partidos de hoy:"
    else:
        sub = df.head(12)
        title = "📅 Elige partido para analizar:"

    await update.message.reply_html(title, reply_markup=fixtures_keyboard(sub))


# ---------------------------------------------------------
# Handlers comandos
# ---------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await _guard(update):
        return

    text = (
        "⚽ <b>GAP FREE v2 Bot</b>\n\n"
        "Ahora puedes analizar sin aprenderte el formato largo.\n\n"
        "Puedes usar botones o escribir algo normal como:\n"
        "<code>brasil japon</code>\n"
        "<code>south africa canada</code>\n"
        "<code>sudafrica canada</code>\n\n"
        "También puedes enviarme un CSV."
    )
    await update.message.reply_html(text, reply_markup=main_menu_keyboard())


async def ayuda(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await _guard(update):
        return

    text = (
        "📌 <b>Formas fáciles de usar el bot</b>\n\n"
        "1) Pulsa <b>Próximos partidos</b> y elige uno.\n\n"
        "2) Escribe el partido sin formato:\n"
        "<code>brasil japon</code>\n"
        "<code>francia suecia</code>\n"
        "<code>south africa canada</code>\n\n"
        "3) Comando manual opcional:\n"
        "<code>/analizar Brazil | Japan | Houston | Dieciseisavos | 2026-06-29 | 7 | 7</code>\n\n"
        "4) Mándame un CSV y te devuelvo el CSV analizado."
    )
    await update.message.reply_html(text, reply_markup=main_menu_keyboard())


async def estado(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await _guard(update):
        return

    ok, problems = _check_local_system()

    if ok:
        text = (
            "✅ <b>Sistema listo</b>\n\n"
            f"Dataset: <code>{PATHS['dataset']}</code>\n"
            f"Team state: <code>{PATHS['team_state']}</code>\n"
            f"Modelo: <code>{PATHS['artifact']}</code>\n\n"
            "Ya puedes analizar con botones o escribiendo el partido."
        )
    else:
        text = "⚠️ <b>Faltan cosas para funcionar:</b>\n\n"
        text += "\n".join(f"• {_safe_text(p)}" for p in problems)

    await update.message.reply_html(text, reply_markup=main_menu_keyboard())


async def proximos(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await _guard(update):
        return
    await send_fixtures(update, context, mode="proximos")


async def hoy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await _guard(update):
        return
    await send_fixtures(update, context, mode="hoy")


async def brasiljapon(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await _guard(update):
        return
    await update.message.chat.send_action(ChatAction.TYPING)
    await analyze_fixture_by_index(update, context, 1)


async def analizar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Mantiene compatibilidad con formato antiguo, pero también acepta:
    /analizar brasil japon
    """
    if not await _guard(update):
        return

    raw = update.message.text.replace("/analizar", "", 1).strip()

    if "|" in raw:
        parts = [p.strip() for p in raw.split("|")]
        if len(parts) < 5:
            await update.message.reply_html(
                "Formato incorrecto. Mejor escribe solo el partido, por ejemplo:\n"
                "<code>brasil japon</code>"
            )
            return

        rows = [{
            "equipo_local": parts[0],
            "equipo_visitante": parts[1],
            "sede": parts[2],
            "fase": parts[3],
            "fecha": parts[4],
            "dias_descanso_local": int(parts[5]) if len(parts) >= 6 and parts[5] else 7,
            "dias_descanso_visitante": int(parts[6]) if len(parts) >= 7 and parts[6] else 7,
        }]

        await update.message.chat.send_action(ChatAction.TYPING)
        out = _run_prediction(rows)
        await update.message.reply_html(_format_prediction_row(out.iloc[0]), reply_markup=main_menu_keyboard())
        return

    # Nuevo modo fácil
    await analizar_texto_libre(update, context, raw)


async def analizar_texto_libre(update: Update, context: ContextTypes.DEFAULT_TYPE, raw: str | None = None):
    if not await _guard(update):
        return

    text = raw if raw is not None else update.message.text
    text = text.strip()

    if not text:
        await update.message.reply_html("Escribe un partido, por ejemplo: <code>brasil japon</code>")
        return

    matches = find_fixture(text, limit=5)

    if matches.empty:
        await update.message.reply_html(
            "No encuentro ese partido.\n\n"
            "Prueba con botones:",
            reply_markup=fixtures_keyboard(load_fixtures().head(12)),
        )
        return

    # Si hay una coincidencia muy clara, analizar directamente.
    if len(matches) == 1 or matches.iloc[0]["_score"] >= 90:
        idx = int(matches.index[0])
        await update.message.chat.send_action(ChatAction.TYPING)
        await analyze_fixture_by_index(update, context, idx)
        return

    # Si hay varias, proponer botones.
    sub = matches.drop(columns=["_score"])
    await update.message.reply_html(
        "He encontrado varias opciones. Elige una:",
        reply_markup=fixtures_keyboard(sub),
    )


# ---------------------------------------------------------
# CSV
# ---------------------------------------------------------

async def documento_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await _guard(update):
        return

    ok, problems = _check_local_system()
    if not ok:
        await update.message.reply_html("⚠️ Primero arregla esto:\n" + "\n".join(f"• {_safe_text(p)}" for p in problems))
        return

    doc = update.message.document
    if not doc:
        return

    if not doc.file_name.lower().endswith(".csv"):
        await update.message.reply_text("Envíame un archivo .csv.")
        return

    await update.message.chat.send_action(ChatAction.TYPING)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    input_path = JOBS_DIR / f"telegram_upload_{ts}_{doc.file_name}"
    output_path = JOBS_DIR / f"telegram_result_{ts}.csv"

    try:
        tg_file = await doc.get_file()
        await tg_file.download_to_drive(custom_path=input_path)

        df = pd.read_csv(input_path)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

        if missing:
            await update.message.reply_html(
                "❌ El CSV no tiene las columnas obligatorias:\n"
                f"<code>{', '.join(missing)}</code>"
            )
            return

        out = predict_batch(input_path, output_path)

        lines = []
        lines.append("📊 <b>Análisis CSV terminado</b>")
        lines.append(f"Partidos analizados: <b>{len(out)}</b>")

        decision_col = "decision" if "decision" in out.columns else "prediccion"
        decisiones = out[decision_col].astype(str).map(_norm_decision)
        valid = out[decisiones.isin(["OVER 2.5", "UNDER 2.5"])]
        lines.append(f"Apuestas validadas: <b>{len(valid)}</b>")
        lines.append("")

        for _, row in out.head(8).iterrows():
            partido = f"{row.get('equipo_local')} vs {row.get('equipo_visitante')}"
            pred = _norm_decision(_row_first(row, "decision", "prediccion", default="NO APOSTAR"))
            obs = _norm_decision(_row_first(row, "observacion_no_apuesta", default=""))
            conf = _pct(_row_first(row, "confianza_apuesta", "confianza_modelo", "confianza_final", default=None))
            lines.append(f"• <b>{_safe_text(partido)}</b>: {html.escape(str(pred))} | obs {html.escape(str(obs))} | conf {conf}")

        if len(out) > 8:
            lines.append(f"\n...y {len(out) - 8} más. Te envío el CSV completo.")

        await update.message.reply_html("\n".join(lines))

        with open(output_path, "rb") as f:
            await update.message.reply_document(
                document=f,
                filename="predicciones_telegram.csv",
                caption="CSV completo con predicciones."
            )

    except Exception:
        err = html.escape(traceback.format_exc())
        await update.message.reply_html(f"❌ Error procesando CSV:\n<pre>{err}</pre>")


# ---------------------------------------------------------
# Callback botones
# ---------------------------------------------------------

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await _guard(update):
        return

    query = update.callback_query
    await query.answer()

    data = query.data or ""

    try:
        if data == "menu:inicio":
            text = (
                "⚽ <b>GAP FREE v2 Bot</b>\n\n"
                "Elige una opción o escribe un partido, por ejemplo:\n"
                "<code>brasil japon</code>"
            )
            await query.edit_message_text(text, parse_mode="HTML", reply_markup=main_menu_keyboard())
            return

        if data == "menu:estado":
            ok, problems = _check_local_system()
            if ok:
                text = "✅ <b>Sistema listo</b>\n\nPuedes analizar partidos."
            else:
                text = "⚠️ <b>Faltan cosas:</b>\n\n" + "\n".join(f"• {_safe_text(p)}" for p in problems)
            await query.edit_message_text(text, parse_mode="HTML", reply_markup=main_menu_keyboard())
            return

        if data == "menu:proximos":
            df = load_fixtures().head(12)
            await query.edit_message_text(
                "📅 Elige partido para analizar:",
                parse_mode="HTML",
                reply_markup=fixtures_keyboard(df),
            )
            return

        if data.startswith("match:"):
            idx = int(data.split(":", 1)[1])
            await query.edit_message_text("⏳ Analizando partido...")
            await analyze_fixture_by_index(query, context, idx)
            return

    except Exception:
        err = html.escape(traceback.format_exc())
        await query.edit_message_text(f"❌ Error:\n<pre>{err}</pre>", parse_mode="HTML")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "Falta TELEGRAM_BOT_TOKEN. En Windows usa:\n"
            "set TELEGRAM_BOT_TOKEN=TU_TOKEN\n"
            "python telegram_bot_free.py"
        )

    # Crea fixtures si no existen.
    if not FIXTURES_FILE.exists():
        _default_fixtures().to_csv(FIXTURES_FILE, index=False, encoding="utf-8")

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ayuda", ayuda))
    app.add_handler(CommandHandler("help", ayuda))
    app.add_handler(CommandHandler("estado", estado))
    app.add_handler(CommandHandler("proximos", proximos))
    app.add_handler(CommandHandler("hoy", hoy))
    app.add_handler(CommandHandler("brasiljapon", brasiljapon))
    app.add_handler(CommandHandler("analizar", analizar))

    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.Document.FileExtension("csv"), documento_csv))

    # Texto libre: "brasil japon", "south africa canada", etc.
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analizar_texto_libre))

    print("Bot fácil iniciado. Pulsa Ctrl+C para parar.")
    print("Puedes escribir en Telegram: brasil japon")
    print("O usar /proximos para botones.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()