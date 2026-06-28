from __future__ import annotations

import argparse
import pandas as pd

from config import PATHS, SETTINGS
from data_sources import load_international_results, try_build_statsbomb
from features import add_elo_goal_form, merge_statsbomb, add_stats_form, build_team_state
from gap_ratings import calculate_gap_ratings
from poisson_models import fit_poisson_config, add_poisson_features


def build_dataset(force: bool = False, no_statsbomb: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    print("[1/6] Cargando resultados internacionales...")
    raw = load_international_results(force=force)
    print(f"  Partidos raw: {len(raw):,}")

    print("[2/6] Calculando ELO + forma de goles + H2H prepartido...")
    df = add_elo_goal_form(raw)
    print(f"  Dataset base: {len(df):,}")

    sb = pd.DataFrame()
    if not no_statsbomb:
        print("[3/6] Enriqueciendo con StatsBomb opcional...")
        sb = try_build_statsbomb(force=force, max_matches=SETTINGS["max_statsbomb_matches"])
        print(f"  StatsBomb: {len(sb):,} partidos con eventos")
    else:
        print("[3/6] StatsBomb desactivado por --no-statsbomb")

    df = merge_statsbomb(df, sb)
    df = add_stats_form(df)

    print("[4/6] Calculando GAP ratings prepartido...")
    df, gap_model = calculate_gap_ratings(df)

    print("[5/6] Calculando Poisson base + Poisson GAP...")
    poisson_config = fit_poisson_config(df)
    df = add_poisson_features(df, poisson_config)

    print("[6/6] Guardando outputs...")
    df = df[pd.to_datetime(df["fecha"]).dt.year >= SETTINGS["train_min_year"]].copy().sort_values("fecha")
    team_state = build_team_state(df)
    df.to_csv(PATHS["dataset"], index=False, encoding="utf-8")
    team_state.to_csv(PATHS["team_state"], index=False, encoding="utf-8")

    report = []
    report.append("OVER25 ULTRA FREE - BUILD REPORT")
    report.append(f"Filas dataset: {len(df):,}")
    report.append(f"Rango fechas: {df['fecha'].min()} -> {df['fecha'].max()}")
    report.append(f"Tasa Over 2.5: {df['over25'].mean():.3f}")
    report.append(f"StatsBomb disponibles: {int(df['statsbomb_available'].sum()):,}")
    report.append(f"Equipos: {team_state['equipo'].nunique() if not team_state.empty else 0}")
    report.append(f"Archivo dataset: {PATHS['dataset']}")
    report.append(f"Archivo team_state: {PATHS['team_state']}")
    PATHS["build_report"].write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return df, team_state, "\n".join(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-statsbomb", action="store_true")
    args = parser.parse_args()
    build_dataset(force=args.force, no_statsbomb=args.no_statsbomb)


if __name__ == "__main__":
    main()
