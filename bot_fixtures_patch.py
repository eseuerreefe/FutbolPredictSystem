"""
bot_fixtures_patch.py
=====================
Añade al telegram_bot_free.py la capacidad de refrescar el CSV
automáticamente sin tocar el código original del bot.

OPCIÓN A — Integración mínima (recomendada):
  Añade estas dos líneas al main() de telegram_bot_free.py, justo antes de
  app.run_polling(...):

      from bot_fixtures_patch import auto_refresh_fixtures, schedule_daily_refresh
      auto_refresh_fixtures()                   # refresca al arrancar
      schedule_daily_refresh(app, hour=7, minute=0)  # refresca cada día a las 7h

OPCIÓN B — Ejecutar independientemente:
  python bot_fixtures_patch.py
  (Esto solo regenera el CSV y sale, útil para cron/Task Scheduler)
"""

from __future__ import annotations

import sys
from datetime import time as dtime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

# Importar el generador
try:
    from fetch_fixtures import build_fixtures, OUTPUT_FILE
except ImportError:
    raise ImportError(
        "No se encuentra fetch_fixtures.py. "
        "Colócalo en la misma carpeta que este archivo."
    )


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIÓN DE REFRESCO
# ──────────────────────────────────────────────────────────────────────────────

def auto_refresh_fixtures(days: int = 14, force: bool = False) -> None:
    """
    Llama a build_fixtures(). Si el CSV ya es reciente (< 6h), no hace nada.
    Llama esto al arrancar el bot.
    """
    print("[bot_fixtures_patch] Verificando fixtures...")
    try:
        df = build_fixtures(days_ahead=days, force=force, all_fixtures=True)
        print(f"[bot_fixtures_patch] ✅ {len(df)} partidos listos en {OUTPUT_FILE.name}")
    except Exception as e:
        print(f"[bot_fixtures_patch] ⚠️  Error al refrescar fixtures: {e}")
        print("  El bot continuará con el CSV existente.")


# ──────────────────────────────────────────────────────────────────────────────
# JOB PERIÓDICO para python-telegram-bot JobQueue
# ──────────────────────────────────────────────────────────────────────────────

def schedule_daily_refresh(app, hour: int = 7, minute: int = 0) -> None:
    """
    Registra un job en la JobQueue de python-telegram-bot para refrescar el CSV
    cada día a la hora indicada (hora local del servidor).

    Uso en main() de telegram_bot_free.py:
        from bot_fixtures_patch import auto_refresh_fixtures, schedule_daily_refresh
        auto_refresh_fixtures()
        schedule_daily_refresh(app, hour=7, minute=0)
    """
    try:
        from telegram.ext import Application
    except ImportError:
        print("[bot_fixtures_patch] python-telegram-bot no encontrado, skip scheduling.")
        return

    async def _job_callback(context):
        print("[bot_fixtures_patch] Job diario — refrescando fixtures...")
        auto_refresh_fixtures(force=True)

    if app.job_queue is None:
        print("[bot_fixtures_patch] ⚠️  JobQueue no disponible. Instala: pip install 'python-telegram-bot[job-queue]'")
        return

    app.job_queue.run_daily(
        _job_callback,
        time=dtime(hour=hour, minute=minute),
        name="daily_fixtures_refresh",
    )
    print(f"[bot_fixtures_patch] ✅ Refresco diario programado a las {hour:02d}:{minute:02d}")


# ──────────────────────────────────────────────────────────────────────────────
# EJECUCIÓN DIRECTA (para cron / Task Scheduler)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Refresca telegram_fixtures_worldcup.csv")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    auto_refresh_fixtures(days=args.days, force=args.force)
