FIX 2 — BACKTEST PROFESIONAL + UMBRALES OPTIMIZADOS
====================================================

Qué cambia:

1) backtest.py deja de ser solo un backtest rápido.
   Ahora hace walk-forward por años:
   - entrena ML solo con años anteriores
   - calibra con tramo temporal anterior
   - predice el año siguiente sin fuga de datos

2) Barre umbrales:
   - confidence_threshold: 0.54 a 0.70
   - max_model_discrepancy: 0.12 a 0.26

3) Guarda:
   outputs/backtest_oos_predictions_ultra.csv
   outputs/backtest_threshold_grid_ultra.csv
   outputs/recommended_thresholds_ultra.json si usas --apply-best

4) predict.py usa automáticamente recommended_thresholds_ultra.json
   si existe y si config.py tiene use_optimized_thresholds=True.

Comandos:

python backtest.py --force
python backtest.py --apply-best
python predict.py "South Africa" "Canada" --sede "Los Angeles" --fase "Dieciseisavos" --fecha "2026-06-28" --dias-local 3 --dias-visitante 3

Si quieres el backtest antiguo rápido:

python backtest.py --quick

Importante:
Esto NO garantiza 100% de acierto. Lo que hace es evitar autoengaños:
elige umbrales con validación temporal fuera de muestra, no a ojo.
