FIX 3B — reparación pequeña

Problema corregido:
- Si backtest.py --apply-best recomienda filter_mode=directional_consensus,
  el JSON puede guardar max_model_discrepancy=null.
- predict.py rompía en filter_certainty.py al hacer float(None).

Cambio:
- filter_certainty.py ahora lee valores numéricos de forma segura.
- max_model_discrepancy=null ya no rompe. En modo directional_consensus no bloquea;
  solo queda como dato informativo/fallback.

No toca:
- dataset
- modelos entrenados
- outputs
- train.py
- build_dataset.py
- backtest.py
- predict.py

Uso:
1) Extraer este ZIP encima de la carpeta actual.
2) Ejecutar directamente predict.py. No hace falta reentrenar ni repetir backtest.
