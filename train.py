from __future__ import annotations

import argparse
import pickle
import pandas as pd

from config import PATHS
from build_dataset import build_dataset
from gap_ratings import calculate_gap_ratings
from poisson_models import fit_poisson_config, add_poisson_features
from ml_model import train_ml


def load_or_build(force_build: bool = False, no_statsbomb: bool = False) -> pd.DataFrame:
    if force_build or not PATHS["dataset"].exists():
        df, _, _ = build_dataset(force=force_build, no_statsbomb=no_statsbomb)
        return df
    df = pd.read_csv(PATHS["dataset"])
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    return df.sort_values("fecha").reset_index(drop=True)


def train(force_build: bool = False, no_statsbomb: bool = False) -> dict:
    print("\n══════════════════════════════════════════════════")
    print("ENTRENAMIENTO OVER25 ULTRA FREE")
    print("══════════════════════════════════════════════════")
    df = load_or_build(force_build, no_statsbomb)
    print(f"Dataset: {len(df):,} filas | {df['fecha'].min()} → {df['fecha'].max()}")
    print(f"Over2.5 rate: {df['over25'].mean():.3f}")
    sb_col = df.get("statsbomb_available", 0)
    sb_sum = int(sb_col.sum()) if hasattr(sb_col, "sum") else int(sb_col)
    print(f"StatsBomb reales: {sb_sum:,}")

    # Recalcular capas por si el dataset venía de versión anterior.
    print("\n[CAPA 0] GAP ratings...")
    df_gap, gap_model = calculate_gap_ratings(df)

    print("[CAPA 1/2] Poisson base + GAP...")
    poisson_config = fit_poisson_config(df_gap)
    df_model = add_poisson_features(df_gap, poisson_config)
    df_model.to_csv(PATHS["dataset"], index=False, encoding="utf-8")

    print("[CAPA 3] ML calibrado temporal...")
    ml = train_ml(df_model)
    if not ml.get("trained"):
        print("  ML no entrenado:", ml.get("reason"))
    else:
        print("  ML entrenado:", ml["report"].get("backend"), "| filas:", ml["report"].get("rows"))

    artifact = {
        "version": "over25_ultra_free_v1",
        "gap_model": gap_model,
        "poisson_config": poisson_config,
        "ml": ml,
    }
    with open(PATHS["artifact"], "wb") as f:
        pickle.dump(artifact, f)

    report = []
    report.append("OVER25 ULTRA FREE - TRAIN REPORT")
    report.append(f"Dataset filas: {len(df_model):,}")
    report.append(f"Over2.5 rate: {df_model['over25'].mean():.3f}")
    report.append(f"Poisson config: {poisson_config}")
    report.append(f"ML trained: {ml.get('trained')}")
    if ml.get("trained"):
        report.append(f"Backend: {ml['report'].get('backend')}")
        report.append(f"Calibración: {ml['report'].get('calibration')}")
    report.append(f"Modelo guardado: {PATHS['artifact']}")
    PATHS["train_report"].write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--no-statsbomb", action="store_true")
    args = parser.parse_args()
    train(force_build=args.force_build, no_statsbomb=args.no_statsbomb)


if __name__ == "__main__":
    main()
