FIX 1 — columnas duplicadas y compatibilidad Python 3.11

Problema corregido:
- Si ejecutabas train.py varias veces sobre data/processed/df_entrenamiento.csv ya enriquecido,
  se recalculaban GAP y Poisson encima de columnas ya existentes.
- Eso creaba columnas duplicadas en memoria.
- En ml_model.py, work[c] podía devolver un DataFrame en vez de una Series y pandas fallaba con:
  TypeError: arg must be a list, tuple, 1-d array, or Series

Archivos corregidos:
- train.py
- ml_model.py
- gap_ratings.py
- poisson_models.py
- features.py
- predict.py

Cambios:
- Se eliminan columnas GAP/Poisson/forma-stats antes de recalcular.
- Se eliminan columnas duplicadas de forma defensiva.
- prepare_matrix ahora es anti-fallos si recibe duplicados.
- predict.py queda compatible con Python 3.11.

Uso recomendado tras copiar los archivos:
  cd /d D:\Mundial\gptsystem3\over25_ultra_free_project
  del models\*.pkl
  python train.py --no-statsbomb
  python predict.py "South Africa" "Canada" --sede "Los Angeles" --fase "Dieciseisavos" --fecha "2026-06-28" --dias-local 3 --dias-visitante 3
