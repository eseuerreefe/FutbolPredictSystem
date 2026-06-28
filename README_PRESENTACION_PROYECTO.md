# Sistema de prediccion Over/Under 2.5 con bot de Telegram

**Memoria tecnica simple para presentar el proyecto a empresas**

## Resumen

Este proyecto es un sistema completo de prediccion futbolistica para el mercado **Over/Under 2.5 goles**. Combina datos historicos, ratings de equipos, modelos estadisticos, machine learning calibrado, reglas de consenso y un bot de Telegram como interfaz.

El objetivo del sistema no es apostar en todos los partidos, sino **filtrar** y recomendar solo cuando los modelos tienen suficiente coherencia. Por eso diferencia entre:

- **Observacion bruta:** lectura del partido.
- **Resultado final:** apuesta validada o NO APOSTAR.

## Arquitectura

| Capa | Archivo | Funcion |
|---|---|---|
| Datos | `data_sources.py`, `build_dataset.py` | Carga y normalizacion de partidos historicos |
| Features | `features.py` | Forma, ELO, descanso, fase, sede y contexto |
| Ratings | `gap_ratings.py` | Fuerza ofensiva/defensiva propia de equipos |
| Estadistica | `poisson_models.py` | Probabilidad base Over/Under mediante Poisson |
| Machine Learning | `ml_model.py` | Modelo calibrado para patrones no lineales |
| Consenso | `consensus.py` | Combinacion de probabilidades |
| Filtro | `filter_certainty.py` | Decide si hay apuesta o NO APOSTAR |
| Prediccion | `predict.py` | CLI y CSV |
| Frontend | `telegram_bot_free.py` | Bot de Telegram con botones |

## Por que puede acertar bien

1. No fuerza picks: si no hay seguridad, devuelve **NO APOSTAR**.
2. Usa varios modelos complementarios, no una unica regla.
3. Separa probabilidad del partido y decision de apuesta.
4. Permite backtesting y evaluacion objetiva.
5. Es local y no depende de APIs de pago para funcionar.

## Comandos principales

Prediccion individual:

```bat
python predict.py "France" "Sweden" --sede "Nueva York" --fase "Dieciseisavos" --fecha "2026-06-30" --dias-local 7 --dias-visitante 7
```

Prediccion por CSV sin reentrenar:

```bat
python -c "from predict import predict_csv; predict_csv(r'INPUT.csv', r'outputs\predicciones.csv')"
```

## Valor profesional

Este proyecto demuestra capacidad para construir una solucion end-to-end:

- Ingenieria de datos.
- Modelizacion estadistica y machine learning.
- Automatizacion.
- Producto usable con Telegram.
- Evaluacion de precision y cobertura.
- Arquitectura modular mantenible.

## Pitch corto

> He desarrollado un sistema completo de prediccion futbolistica en Python para Over/Under 2.5. Incluye dataset, features, ratings, Poisson, machine learning calibrado, consenso, filtros de certeza, CSV y bot de Telegram. El valor del proyecto es que demuestra una solucion real de principio a fin: datos, modelo, automatizacion, producto y evaluacion objetiva.
