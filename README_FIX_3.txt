FIX 3 — CONSENSO DIRECCIONAL 45/55
====================================

Objetivo:
No perder nada útil del FIX 2, pero corregir el filtro que bloqueaba apuestas solo
porque la discrepancia absoluta entre modelos era mayor que 0.12.

Antes:
- Si max(modelos) - min(modelos) > 0.12 => NO APOSTAR.
- Esto bloqueaba casos donde todos los modelos iban hacia el mismo lado,
  por ejemplo 28%, 41%, 44.9% Over => todos apuntan UNDER, pero diferencia = 16.4%.

Ahora:
- Modo principal: directional_consensus.
- Para apostar UNDER 2.5:
  todos los modelos disponibles deben tener P(Over) < 45%.
- Para apostar OVER 2.5:
  todos los modelos disponibles deben tener P(Over) >= 55%.
- La discrepancia absoluta se conserva como información y para comparación legacy,
  pero ya no bloquea por defecto.

Archivos corregidos:
- config.py
- filter_certainty.py
- backtest.py
- predict.py

No se tocan:
- dataset
- modelos entrenados
- outputs históricos
- arquitectura base
- LightGBM
- Poisson base
- Poisson GAP
- consenso de probabilidades

Cómo instalar encima sin perder datos:
1) Cierra CMD si está ejecutando algo.
2) Extrae este ZIP en la carpeta padre donde tienes over25_ultra_free_project_FIX2_src.
   Ejemplo:
   cd /d D:\Mundial\gptsystem4
   powershell -Command "Expand-Archive -Path $env:USERPROFILE\Downloads\over25_ultra_free_project_FIX3.zip -DestinationPath D:\Mundial\gptsystem4 -Force"

3) Entra en el proyecto:
   cd /d D:\Mundial\gptsystem4\over25_ultra_free_project_FIX2_src

4) No hace falta reconstruir dataset ni reentrenar solo por el filtro:
   python predict.py "South Africa" "Canada" --sede "Los Angeles" --fase "Dieciseisavos" --fecha "2026-06-28" --dias-local 3 --dias-visitante 3

5) Recomendado: recalcular el backtest FIX3 y aplicar mejores umbrales:
   python backtest.py --force
   python backtest.py --apply-best

Notas:
- Si existe outputs/recommended_thresholds_ultra.json antiguo del FIX 2, predict.py
  usará igualmente directional_consensus por defecto salvo que el JSON nuevo indique otro modo.
- backtest.py ahora compara tres modos:
  1. directional_consensus  (nuevo recomendado)
  2. legacy_discrepancy     (antiguo, conservado para comparar)
  3. hybrid_both            (ambos filtros a la vez)

Ejemplo de interpretación:
- 28.5%, 44.9%, 41.1% Over => UNDER permitido si la confianza final supera umbral.
- 33%, 45.0%, 41% Over => bloquea UNDER si el valor real llega a 45.0% o más.
- 62%, 53%, 61% Over => bloquea OVER porque un modelo no llega a 55%.
