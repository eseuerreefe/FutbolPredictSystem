"""
telegram_fixtures_manager.py
============================
Añade al bot de Telegram la capacidad de gestionar el CSV de partidos
directamente desde Telegram, sin tocar archivos manualmente.

COMANDOS NUEVOS:
  /actualizar        → Descarga partidos de los próximos 14 días y regenera el CSV
  /actualizar 30     → Próximos 30 días
  /partidos_hoy      → Solo partidos de hoy
  /añadir            → Flujo guiado para añadir un partido manualmente
  /borrar            → Lista partidos con botón de borrar
  /ver_csv           → Te envía el CSV actual como archivo descargable

INTEGRACIÓN CON telegram_bot_free.py:
  Añade al final de main(), antes de app.run_polling():

      from telegram_fixtures_manager import register_fixture_handlers
      register_fixture_handlers(app)

  Y listo. El resto del bot sigue igual.

SEGURIDAD:
  Solo usuarios en ALLOWED_TELEGRAM_USERS pueden usar /actualizar, /añadir y /borrar.
  /ver_csv también está restringido.
"""

from __future__ import annotations

import html
import os
import sys
import traceback
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

FIXTURES_FILE = PROJECT_DIR / "telegram_fixtures_worldcup.csv"

# Columnas que espera el bot
REQUIRED_COLUMNS = [
    "game_id", "equipo_local", "equipo_visitante",
    "sede", "fase", "fecha", "hora_espana",
    "dias_descanso_local", "dias_descanso_visitante",
]

FASES_VALIDAS = ["Grupo", "Dieciseisavos", "Octavos", "Cuartos", "Semis", "Final", "Tercer_puesto"]

try:
    from telegram import (
        Update, InlineKeyboardButton, InlineKeyboardMarkup,
    )
    from telegram.constants import ChatAction, ParseMode
    from telegram.ext import (
        Application, CommandHandler, CallbackQueryHandler,
        MessageHandler, ConversationHandler, ContextTypes, filters,
    )
except ImportError:
    raise ImportError("Instala: pip install 'python-telegram-bot>=20.0'")

# ──────────────────────────────────────────────────────────────────────────────
# ESTADOS ConversationHandler para /añadir
# ──────────────────────────────────────────────────────────────────────────────
ADD_LOCAL, ADD_VISITANTE, ADD_SEDE, ADD_FASE, ADD_FECHA, ADD_HORA, ADD_CONFIRM = range(7)


# ──────────────────────────────────────────────────────────────────────────────
# SEGURIDAD
# ──────────────────────────────────────────────────────────────────────────────

def _allowed(user_id: int) -> bool:
    allowed = os.getenv("ALLOWED_TELEGRAM_USERS", "").strip()
    if not allowed:
        return True
    return str(user_id) in {x.strip() for x in allowed.split(",") if x.strip()}


async def _guard(update: Update) -> bool:
    user = update.effective_user
    if user and not _allowed(user.id):
        await update.effective_message.reply_text("⛔ No estás autorizado.")
        return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS CSV
# ──────────────────────────────────────────────────────────────────────────────

def _load_csv() -> pd.DataFrame:
    if not FIXTURES_FILE.exists():
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    df = pd.read_csv(FIXTURES_FILE)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col not in ("dias_descanso_local", "dias_descanso_visitante") else 7
    return df.reset_index(drop=True)


def _save_csv(df: pd.DataFrame) -> None:
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col not in ("dias_descanso_local", "dias_descanso_visitante") else 7
    df[REQUIRED_COLUMNS].to_csv(FIXTURES_FILE, index=False, encoding="utf-8")


def _next_game_id(df: pd.DataFrame) -> str:
    numeric = pd.to_numeric(df["game_id"], errors="coerce").dropna()
    return str(int(numeric.max()) + 1) if len(numeric) else "1000"


def _days_rest(team: str, df: pd.DataFrame, match_date: str) -> int:
    if df.empty or not team:
        return 7
    mask = (df["equipo_local"].astype(str) == team) | (df["equipo_visitante"].astype(str) == team)
    prev = df[mask].copy()
    if prev.empty:
        return 7
    prev["fecha"] = pd.to_datetime(prev["fecha"], errors="coerce")
    target = pd.to_datetime(match_date, errors="coerce")
    if pd.isna(target):
        return 7
    antes = prev[prev["fecha"] < target].sort_values("fecha")
    if antes.empty:
        return 7
    return max(1, min(int((target - antes.iloc[-1]["fecha"]).days), 30))


# ──────────────────────────────────────────────────────────────────────────────
# /actualizar — Descarga automática desde fetch_fixtures.py
# ──────────────────────────────────────────────────────────────────────────────

async def cmd_actualizar(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _guard(update):
        return

    # Parsear argumento opcional: /actualizar 30
    args = ctx.args or []
    try:
        days = int(args[0]) if args else 14
        days = max(1, min(days, 90))
    except ValueError:
        days = 14

    msg = await update.message.reply_text(
        f"⏳ Descargando partidos de los próximos <b>{days} días</b>...",
        parse_mode=ParseMode.HTML,
    )

    try:
        from fetch_fixtures import build_fixtures
        df = build_fixtures(days_ahead=days, force=True, all_fixtures=(days >= 60))

        # Resumen por fases
        resumen_fases = ""
        if "fase" in df.columns:
            counts = df["fase"].value_counts()
            resumen_fases = "\n".join(f"  • {fase}: {n}" for fase, n in counts.items())

        # Próximos 5 partidos
        hoy = date.today().isoformat()
        proximos = df[df["fecha"].astype(str) >= hoy].head(5)
        proximos_txt = ""
        for _, r in proximos.iterrows():
            hora = str(r.get("hora_espana", "") or "")
            proximos_txt += f"\n  📅 {r['fecha']} {hora} · <b>{html.escape(str(r['equipo_local']))}</b> vs <b>{html.escape(str(r['equipo_visitante']))}</b> ({r['fase']})"

        await msg.edit_text(
            f"✅ <b>CSV actualizado con {len(df)} partidos</b>\n\n"
            f"<b>Por fase:</b>\n{resumen_fases}\n\n"
            f"<b>Próximos partidos:</b>{proximos_txt}\n\n"
            f"Usa /proximos para verlos con botones.",
            parse_mode=ParseMode.HTML,
        )

    except ImportError:
        await msg.edit_text(
            "⚠️ No se encuentra <code>fetch_fixtures.py</code>.\n"
            "Colócalo en la misma carpeta que el bot.",
            parse_mode=ParseMode.HTML,
        )
    except Exception:
        err = html.escape(traceback.format_exc(limit=6))
        await msg.edit_text(
            f"❌ Error al actualizar:\n<pre>{err}</pre>",
            parse_mode=ParseMode.HTML,
        )


# ──────────────────────────────────────────────────────────────────────────────
# /ver_csv — Envía el CSV actual como archivo
# ──────────────────────────────────────────────────────────────────────────────

async def cmd_ver_csv(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _guard(update):
        return

    df = _load_csv()
    if df.empty:
        await update.message.reply_text("⚠️ El CSV está vacío. Usa /actualizar primero.")
        return

    hoy = date.today().isoformat()
    futuros = df[df["fecha"].astype(str) >= hoy]
    pasados = df[df["fecha"].astype(str) < hoy]

    caption = (
        f"📋 <b>telegram_fixtures_worldcup.csv</b>\n"
        f"Total: <b>{len(df)}</b> partidos\n"
        f"Pendientes: <b>{len(futuros)}</b>  |  Jugados: <b>{len(pasados)}</b>\n"
        f"Rango: {df['fecha'].min()} → {df['fecha'].max()}"
    )

    with open(FIXTURES_FILE, "rb") as f:
        await update.message.reply_document(
            document=f,
            filename="telegram_fixtures_worldcup.csv",
            caption=caption,
            parse_mode=ParseMode.HTML,
        )


# ──────────────────────────────────────────────────────────────────────────────
# /borrar — Lista partidos con botón de borrar
# ──────────────────────────────────────────────────────────────────────────────

async def cmd_borrar(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _guard(update):
        return

    df = _load_csv()
    hoy = date.today().isoformat()
    futuros = df[df["fecha"].astype(str) >= hoy].copy()

    if futuros.empty:
        await update.message.reply_text("No hay partidos futuros en el CSV.")
        return

    buttons = []
    for idx in futuros.index:
        r = df.loc[idx]
        label = f"🗑 {r['fecha']} · {r['equipo_local']} vs {r['equipo_visitante']}"
        if len(label) > 60:
            label = label[:57] + "..."
        buttons.append([InlineKeyboardButton(label, callback_data=f"del_fixture:{idx}")])

    await update.message.reply_text(
        f"🗑 <b>Selecciona el partido a borrar</b> ({len(futuros)} pendientes):",
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(buttons),
    )


async def callback_borrar(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if not _allowed(query.from_user.id):
        await query.answer("⛔ No autorizado.", show_alert=True)
        return

    idx = int(query.data.split(":")[1])
    df = _load_csv()

    if idx not in df.index:
        await query.edit_message_text("⚠️ Partido no encontrado (¿ya borrado?).")
        return

    r = df.loc[idx]
    partido = f"{r['equipo_local']} vs {r['equipo_visitante']} ({r['fecha']})"
    df = df.drop(index=idx).reset_index(drop=True)
    _save_csv(df)

    await query.edit_message_text(
        f"🗑 Borrado: <b>{html.escape(partido)}</b>\n"
        f"Quedan {len(df)} partidos en el CSV.",
        parse_mode=ParseMode.HTML,
    )


# ──────────────────────────────────────────────────────────────────────────────
# /añadir — Flujo conversacional para añadir un partido manual
# ──────────────────────────────────────────────────────────────────────────────

def _fase_keyboard() -> InlineKeyboardMarkup:
    buttons = [[InlineKeyboardButton(f, callback_data=f"add_fase:{f}")] for f in FASES_VALIDAS]
    return InlineKeyboardMarkup(buttons)


async def cmd_anadir(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _guard(update):
        return ConversationHandler.END
    ctx.user_data.clear()
    await update.message.reply_text(
        "➕ <b>Añadir partido</b>\n\nEscribe el <b>equipo local</b>:\n/cancelar para salir",
        parse_mode=ParseMode.HTML,
    )
    return ADD_LOCAL


async def add_local(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data["add_local"] = update.message.text.strip()
    await update.message.reply_text("✏️ Equipo <b>visitante</b>:", parse_mode=ParseMode.HTML)
    return ADD_VISITANTE


async def add_visitante(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data["add_visitante"] = update.message.text.strip()
    await update.message.reply_text(
        "✏️ <b>Sede</b> (ciudad/estadio):\n<i>Escribe <code>-</code> para dejar en blanco</i>",
        parse_mode=ParseMode.HTML,
    )
    return ADD_SEDE


async def add_sede(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    sede = update.message.text.strip()
    ctx.user_data["add_sede"] = "" if sede == "-" else sede
    await update.message.reply_text("✏️ Selecciona la <b>fase</b>:", parse_mode=ParseMode.HTML, reply_markup=_fase_keyboard())
    return ADD_FASE


async def add_fase(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    ctx.user_data["add_fase"] = query.data.replace("add_fase:", "")
    await query.edit_message_text(
        f"Fase: <b>{ctx.user_data['add_fase']}</b>\n\n✏️ <b>Fecha</b> del partido (formato YYYY-MM-DD):",
        parse_mode=ParseMode.HTML,
    )
    return ADD_FECHA


async def add_fecha(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    fecha = update.message.text.strip()
    # Validar formato
    try:
        date.fromisoformat(fecha)
    except ValueError:
        await update.message.reply_text(
            "⚠️ Formato incorrecto. Escribe la fecha como <code>2026-07-10</code>:",
            parse_mode=ParseMode.HTML,
        )
        return ADD_FECHA
    ctx.user_data["add_fecha"] = fecha
    await update.message.reply_text(
        "✏️ <b>Hora en España</b> (HH:MM):\n<i>Escribe <code>-</code> si no la sabes</i>",
        parse_mode=ParseMode.HTML,
    )
    return ADD_HORA


async def add_hora(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    hora = update.message.text.strip()
    ctx.user_data["add_hora"] = "" if hora == "-" else hora

    # Calcular días de descanso automáticamente
    df = _load_csv()
    local    = ctx.user_data["add_local"]
    visitante = ctx.user_data["add_visitante"]
    fecha    = ctx.user_data["add_fecha"]
    rest_l = _days_rest(local,    df, fecha)
    rest_v = _days_rest(visitante, df, fecha)
    ctx.user_data["add_rest_l"] = rest_l
    ctx.user_data["add_rest_v"] = rest_v

    # Resumen para confirmar
    resumen = (
        f"📋 <b>Confirma el partido:</b>\n\n"
        f"⚽ <b>{html.escape(local)}</b> vs <b>{html.escape(visitante)}</b>\n"
        f"📅 Fecha: <b>{fecha}</b>  {ctx.user_data['add_hora']}\n"
        f"🏟 Sede: <b>{html.escape(ctx.user_data['add_sede'] or '—')}</b>\n"
        f"🏆 Fase: <b>{ctx.user_data['add_fase']}</b>\n"
        f"😴 Descanso: local {rest_l}d  |  visitante {rest_v}d"
    )
    kb = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Confirmar", callback_data="add_confirm:si"),
            InlineKeyboardButton("❌ Cancelar",  callback_data="add_confirm:no"),
        ]
    ])
    await update.message.reply_html(resumen, reply_markup=kb)
    return ADD_CONFIRM


async def add_confirm(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == "add_confirm:no":
        await query.edit_message_text("❌ Partido no añadido.")
        ctx.user_data.clear()
        return ConversationHandler.END

    df = _load_csv()
    new_row = {
        "game_id":                  _next_game_id(df),
        "equipo_local":             ctx.user_data["add_local"],
        "equipo_visitante":         ctx.user_data["add_visitante"],
        "sede":                     ctx.user_data.get("add_sede", ""),
        "fase":                     ctx.user_data["add_fase"],
        "fecha":                    ctx.user_data["add_fecha"],
        "hora_espana":              ctx.user_data.get("add_hora", ""),
        "dias_descanso_local":      ctx.user_data.get("add_rest_l", 7),
        "dias_descanso_visitante":  ctx.user_data.get("add_rest_v", 7),
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df = df.sort_values("fecha").reset_index(drop=True)
    _save_csv(df)

    local    = new_row["equipo_local"]
    visitante = new_row["equipo_visitante"]
    await query.edit_message_text(
        f"✅ Partido añadido: <b>{html.escape(local)} vs {html.escape(visitante)}</b> ({new_row['fecha']})\n"
        f"Total en CSV: {len(df)} partidos.\n\n"
        f"Usa /proximos para verlo en el bot.",
        parse_mode=ParseMode.HTML,
    )
    ctx.user_data.clear()
    return ConversationHandler.END


async def add_cancelar(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data.clear()
    await update.message.reply_text("❌ Operación cancelada.")
    return ConversationHandler.END


# ──────────────────────────────────────────────────────────────────────────────
# REGISTRO DE HANDLERS — llamar desde main() del bot
# ──────────────────────────────────────────────────────────────────────────────

def register_fixture_handlers(app: "Application") -> None:
    """
    Registra todos los handlers de gestión de fixtures.
    Llama esto en main() de telegram_bot_free.py antes de run_polling():

        from telegram_fixtures_manager import register_fixture_handlers
        register_fixture_handlers(app)
    """

    # ConversationHandler para /añadir
    conv_add = ConversationHandler(
        entry_points=[
            CommandHandler("anadir",  cmd_anadir),
            CommandHandler("añadir",  cmd_anadir),
            CommandHandler("add",     cmd_anadir),
        ],
        states={
            ADD_LOCAL:     [MessageHandler(filters.TEXT & ~filters.COMMAND, add_local)],
            ADD_VISITANTE: [MessageHandler(filters.TEXT & ~filters.COMMAND, add_visitante)],
            ADD_SEDE:      [MessageHandler(filters.TEXT & ~filters.COMMAND, add_sede)],
            ADD_FASE:      [CallbackQueryHandler(add_fase, pattern=r"^add_fase:")],
            ADD_FECHA:     [MessageHandler(filters.TEXT & ~filters.COMMAND, add_fecha)],
            ADD_HORA:      [MessageHandler(filters.TEXT & ~filters.COMMAND, add_hora)],
            ADD_CONFIRM:   [CallbackQueryHandler(add_confirm, pattern=r"^add_confirm:")],
        },
        fallbacks=[
            CommandHandler("cancelar", add_cancelar),
            CommandHandler("cancel",   add_cancelar),
        ],
        allow_reentry=True,
    )

    app.add_handler(CommandHandler("actualizar",   cmd_actualizar))
    app.add_handler(CommandHandler("ver_csv",      cmd_ver_csv))
    app.add_handler(CommandHandler("borrar",       cmd_borrar))
    app.add_handler(conv_add)
    app.add_handler(CallbackQueryHandler(callback_borrar, pattern=r"^del_fixture:"))

    print("[telegram_fixtures_manager] ✅ Handlers registrados:")
    print("   /actualizar [días]  → descarga partidos automáticamente")
    print("   /añadir             → añade partido manualmente")
    print("   /borrar             → borra un partido")
    print("   /ver_csv            → descarga el CSV actual")
