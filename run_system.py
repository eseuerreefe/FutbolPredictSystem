from __future__ import annotations

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="CSV de partidos para predecir")
    parser.add_argument("--output", default="outputs/predicciones_ultra.csv")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--no-statsbomb", action="store_true")
    args = parser.parse_args()

    build_cmd = [sys.executable, "build_dataset.py"]
    if args.force_build:
        build_cmd.append("--force")
    if args.no_statsbomb:
        build_cmd.append("--no-statsbomb")

    print("\n[1/3] Construyendo dataset...")
    subprocess.check_call(build_cmd)

    train_cmd = [sys.executable, "train.py"]
    if args.no_statsbomb:
        train_cmd.append("--no-statsbomb")
    print("\n[2/3] Entrenando modelo...")
    subprocess.check_call(train_cmd)

    if args.input:
        print("\n[3/3] Prediciendo CSV...")
        subprocess.check_call([sys.executable, "predict.py", "--input", args.input, "--output", args.output])
        print("\nHecho:", args.output)
    else:
        print("\n[3/3] No se indicó --input. Modelo listo para usar con predict.py")


if __name__ == "__main__":
    main()
