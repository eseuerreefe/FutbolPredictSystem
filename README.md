# OVER25 ULTRA FREE

Sistema de predicción Over/Under 2.5 para fútbol internacional.

## Qué es

Este proyecto une lo mejor del modelo simple que te funcionó con la arquitectura compleja del Prompt 2:

1. **Base robusta**: ELO propio + forma de goles + ritmo reciente + descanso + fase.
2. **StatsBomb opcional**: tiros/xG reales cuando existen, sin bloquear cuando faltan.
3. **GAP ratings**: ratings de proceso con tiros reales si existen y fallback por goles si no.
4. **Poisson doble**: Poisson base estable + Poisson GAP avanzado.
5. **ML calibrado**: LightGBM si está instalado; si no, fallback de sklearn.
6. **Consenso final**: mezcla Poisson base + Poisson GAP + ML, en vez de dejar que se contradigan sin control.
7. **Filtro de certeza**: si la señal es floja o contradictoria, devuelve `NO APOSTAR`.

No existe un modelo 100% seguro en apuestas. Este proyecto está diseñado para **no romperse**, ser conservador y evitar falsos 100%.

## Instalación

```bat
cd /d D:\Mundial\over25_ultra_free
pip install -r requirements.txt
```

Si LightGBM te da problemas, el sistema puede funcionar con sklearn, pero es mejor instalarlo:

```bat
pip install lightgbm
```

## Uso rápido

Construir dataset:

```bat
python build_dataset.py
```

Si quieres ir más rápido y saltar StatsBomb:

```bat
python build_dataset.py --no-statsbomb
```

Entrenar:

```bat
python train.py
```

Predecir un partido:

```bat
python predict.py "South Africa" "Canada" --sede "Los Angeles" --fase "Dieciseisavos" --fecha "2026-06-28" --dias-local 3 --dias-visitante 3
```

Predecir un CSV:

```bat
python predict.py --input partidos_ejemplo.csv --output outputs\predicciones_ultra.csv
```

Ejecutar todo:

```bat
python run_system.py --input partidos_ejemplo.csv
```

## Salida esperada

El sistema muestra:

- Poisson base Over
- Poisson GAP Over
- ML calibrado Over
- Probabilidad final Over/Under
- Confianza del modelo
- Confianza de apuesta
- Bloqueos del filtro

Si devuelve `NO APOSTAR`, no es fallo: significa que no ve ventaja suficiente.

## Diferencia con tu modelo anterior

El modelo simple acertaba porque usaba señales robustas: ELO, forma y Poisson simple. Este proyecto conserva eso como columna vertebral.

El modelo complejo fallaba porque se apoyaba demasiado en tiros/corners imputados. Este proyecto:

- No trata datos imputados como si fueran reales.
- No fuerza LightGBM a contradecir Poisson.
- No usa calibración isotónica con pocos datos.
- No pone varianzas a 0 artificialmente.
- No muestra 100% salvo que realmente la calibración lo justifique, y aun así lo recorta entre 0.1% y 99.9%.

## Archivos principales

- `build_dataset.py`: crea dataset histórico.
- `train.py`: entrena GAP, Poisson y ML.
- `predict.py`: predice partidos.
- `backtest.py`: backtest rápido del filtro.
- `config.py`: todos los parámetros.
- `gap_ratings.py`: capa GAP.
- `poisson_models.py`: Poisson base y GAP.
- `ml_model.py`: LightGBM/fallback calibrado.
- `consensus.py`: mezcla final.
- `filter_certainty.py`: filtro de apuesta.
