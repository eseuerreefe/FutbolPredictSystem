# FutbolPredictSystem — OVER25 ULTRA FREE

Sistema de predicción **Over/Under 2.5 goles** para partidos de fútbol internacional (con foco en Mundial), con salida integrada vía bot de Telegram.

> **Nota sobre la naturaleza del sistema:** este proyecto **no es una red neuronal**. Es un **sistema ensemble** que combina un modelo estadístico clásico (Poisson), un modelo de *gradient boosting* (LightGBM, con fallback a scikit-learn) y una capa de reglas (rating ELO propio, filtro de certeza). Ninguno de esos componentes es una red neuronal — LightGBM entrena árboles de decisión, no capas de neuronas. Es importante decirlo así de claro porque es un diseño **más robusto y explicable** que una red neuronal para este caso (poco volumen de datos históricos, necesidad de auditar por qué el sistema dice lo que dice, y cero tolerancia a "cajas negras" cuando hay decisiones de apuesta de por medio).

---

## 1. Qué hace el sistema

Dado un partido (o una lista de partidos en CSV), el sistema estima la probabilidad de que se marquen **más o menos de 2.5 goles** y devuelve una recomendación con su nivel de confianza — o `NO APOSTAR` si la señal es débil o contradictoria.

El diseño prioriza **no romperse y no sobreestimar certeza** por encima de maximizar aciertos a toda costa: prefiere abstenerse antes que dar un 100% falso.

## 2. Arquitectura — pipeline de decisión

El sistema no usa un único modelo, sino que combina varias fuentes de señal en capas, y solo al final las mezcla:

```
                    ┌─────────────────────┐
                    │   Datos de partido   │
                    │ (equipos, fecha,     │
                    │  sede, fase, etc.)   │
                    └──────────┬───────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                       ▼
 ┌──────────────┐     ┌────────────────┐      ┌────────────────┐
 │  ELO propio   │     │  GAP ratings   │      │ StatsBomb (opt) │
 │ forma/ritmo/  │     │ (proceso: xG/  │      │ tiros/xG reales │
 │ descanso/fase │     │ tiros o goles) │      │ si están         │
 └───────┬───────┘     └───────┬────────┘      └────────┬────────┘
         │                     │                        │
         ▼                     ▼                        ▼
 ┌────────────────┐   ┌─────────────────┐      ┌─────────────────┐
 │ Poisson base    │   │  Poisson GAP    │      │  ML calibrado   │
 │ (estable)       │   │  (avanzado)     │      │ LightGBM/sklearn│
 └────────┬────────┘   └────────┬────────┘      └────────┬────────┘
          │                     │                         │
          └───────────┬─────────┴────────────┬────────────┘
                       ▼                      ▼
              ┌─────────────────┐    ┌──────────────────┐
              │   Consenso final │───▶│ Filtro de certeza │
              │ (mezcla ponderada)│    │ (bloquea señales  │
              └─────────────────┘    │  flojas/contrad.) │
                                      └─────────┬─────────┘
                                                ▼
                                  Predicción final: Over / Under /
                                       NO APOSTAR + confianza
```

**Por qué en capas y no un solo modelo:** cada capa cubre el punto débil de la anterior. El Poisson base nunca falla porque solo necesita goles históricos. El GAP (proceso, no resultado) y StatsBomb añaden precisión cuando hay datos de tiros/xG reales, pero **no se activan a la fuerza** si esos datos no existen o están imputados. El ML aporta no-linealidad, pero está calibrado para no contradecir al Poisson sin motivo. El consenso evita que un solo modelo "raro" decida el partido, y el filtro de certeza es el que realmente protege del sobreajuste: si las señales no coinciden, el sistema **se calla** en vez de arriesgar.

## 3. Componentes del pipeline (mapa de archivos)

| Etapa | Archivo | Qué hace |
|---|---|---|
| Datos | `data_sources.py`, `fetch_fixtures.py`, `bot_fixtures_patch.py` | Obtención de partidos/fixtures (incl. integraciones con Telegram) |
| Construcción de dataset | `build_dataset.py` | Genera el histórico de entrenamiento (con o sin StatsBomb, flag `--no-statsbomb`) |
| Features | `features.py` | ELO, forma de goles, ritmo reciente, descanso, fase del torneo |
| Rating de proceso | `gap_ratings.py` | GAP ratings (tiros/xG reales con fallback a goles) |
| Modelos probabilísticos | `poisson_models.py` | Poisson base + Poisson GAP |
| Modelo ML | `ml_model.py` | LightGBM calibrado, con fallback a scikit-learn si LightGBM no está disponible |
| Mezcla final | `consensus.py` | Combina Poisson base + Poisson GAP + ML en una única probabilidad |
| Control de riesgo | `filter_certainty.py` | Bloquea la apuesta (`NO APOSTAR`) si la señal es floja o contradictoria |
| Entrenamiento | `train.py` | Entrena GAP, Poisson y ML sobre el dataset construido |
| Predicción | `predict.py` | Predice un partido suelto o un CSV de partidos |
| Validación | `backtest.py` | Backtest rápido del filtro de certeza sobre histórico |
| Orquestador | `run_system.py` | Ejecuta el flujo completo de principio a fin |
| Configuración | `config.py` | Todos los parámetros del sistema en un solo sitio |
| Utilidades | `utils.py` | Funciones auxiliares comunes |
| Bot de Telegram | `telegram_bot_free.py`, `telegram_fixtures_manager.py` | Interfaz de usuario vía Telegram: consulta de predicciones y próximos partidos |
| Datos de referencia | `partidos_mundial.csv`, `partidos_ejemplo.csv`, `telegram_fixtures_worldcup.csv`, `pm.csv` | Datasets de partidos usados por el sistema |

## 4. Instalación

```bat
cd /d D:\Mundial\over25_ultra_free
pip install -r requirements.txt
```

LightGBM es opcional pero recomendado (si falla, el sistema usa scikit-learn automáticamente):

```bat
pip install lightgbm
```

## 5. Uso

**1. Construir el dataset histórico:**
```bat
python build_dataset.py
```
Para saltar la integración con StatsBomb (más rápido, menos preciso):
```bat
python build_dataset.py --no-statsbomb
```

**2. Entrenar los modelos:**
```bat
python train.py
```

**3. Predecir un partido concreto:**
```bat
python predict.py "South Africa" "Canada" --sede "Los Angeles" --fase "Dieciseisavos" --fecha "2026-06-28" --dias-local 3 --dias-visitante 3
```

**4. Predecir un CSV de partidos:**
```bat
python predict.py --input partidos_ejemplo.csv --output outputs\predicciones_ultra.csv
```

**5. Ejecutar el flujo completo:**
```bat
python run_system.py --input partidos_ejemplo.csv
```

## 6. Salida del sistema

Para cada partido, el sistema muestra:

- Probabilidad Over según Poisson base
- Probabilidad Over según Poisson GAP
- Probabilidad Over según ML calibrado
- Probabilidad final Over/Under (consenso)
- Confianza del modelo
- Confianza de apuesta
- Bloqueos aplicados por el filtro de certeza (si los hay)

Si la salida es `NO APOSTAR`, **no es un fallo del sistema**: significa que no detecta ventaja suficiente para recomendar una posición.

## 7. Decisiones de diseño deliberadas

Estas son restricciones que el sistema respeta a propósito, para evitar los errores típicos de este tipo de proyectos:

- No trata datos imputados (estimados) como si fueran datos reales.
- No deja que LightGBM contradiga al Poisson sin control.
- No aplica calibración isotónica cuando hay pocos datos (sobreajusta).
- No fuerza varianzas a cero artificialmente.
- No muestra un 100% de confianza salvo que la calibración lo justifique realmente, y aun así el valor se recorta entre 0.1% y 99.9%.

## 8. Limitaciones

- No existe ningún modelo 100% fiable en apuestas deportivas; este sistema no lo pretende.
- La calidad de las predicciones depende de la disponibilidad de datos de StatsBomb (tiros/xG reales); sin ellos, el sistema degrada de forma controlada al fallback por goles, pero con menos precisión.
- Pensado y ajustado para competiciones internacionales tipo Mundial; su rendimiento en ligas domésticas no está validado del mismo modo.

## 9. Estructura del repositorio

```
data/                                  # Datos crudos/procesados
fotos/                                 # Recursos visuales
models/                                # Modelos entrenados serializados
outputs/                               # Predicciones generadas
over25_ultra_free_project_FIX2_src/    # Código fuente (versión de trabajo)
__init__.py
config.py
data_sources.py
build_dataset.py
features.py
gap_ratings.py
poisson_models.py
ml_model.py
consensus.py
filter_certainty.py
train.py
predict.py
run_system.py
backtest.py
utils.py
fetch_fixtures.py
bot_fixtures_patch.py
telegram_bot_free.py
telegram_fixtures_manager.py
requirements.txt
```

## 10. Roadmap / estado

Proyecto en desarrollo activo, construido de forma iterativa (asistido por IA + ajustes manuales). Ver `Memoria_tecnica_bot_prediccion_over25_estilo...` para el detalle técnico completo del proceso de construcción.
