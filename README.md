# FutbolPredictSystem — OVER 2.5

Over/Under 2.5 goals prediction system for international football matches (World Cup–focused), with output delivered through a Telegram bot.

> **A note on what this system actually is:** this project is **not a neural network**. It's an **ensemble system** that combines a classical statistical model (Poisson), a gradient boosting model (LightGBM, with a scikit-learn fallback), and a rules layer (a custom ELO rating, a certainty filter). None of those components is a neural network — LightGBM trains decision trees, not layers of neurons. This is worth stating clearly, because this design is actually **more robust and explainable** than a neural network for this use case (limited historical data, the need to audit why the system says what it says, and zero tolerance for "black boxes" when betting decisions are involved).

---

## 1. What the system does

Given a match (or a list of matches in a CSV), the system estimates the probability that the match will finish **over or under 2.5 goals**, and returns a recommendation with a confidence level — or `NO APOSTAR` (DO NOT BET) if the signal is weak or contradictory.

The design prioritizes **not breaking and not overstating certainty** over maximizing hit rate at all costs: it would rather abstain than return a false 100%.

## 2. Architecture — decision pipeline

The system doesn't rely on a single model. It combines several signal sources in layers, and only mixes them at the very end:

```
                    ┌─────────────────────┐
                    │     Match data       │
                    │ (teams, date, venue,  │
                    │  stage, etc.)         │
                    └──────────┬───────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                       ▼
 ┌──────────────┐     ┌────────────────┐      ┌────────────────┐
 │  Custom ELO   │     │  GAP ratings   │      │ StatsBomb (opt) │
 │ form/pace/    │     │ (process: xG/  │      │ real shots/xG   │
 │ rest/stage    │     │ shots or goals)│      │ when available  │
 └───────┬───────┘     └───────┬────────┘      └────────┬────────┘
         │                     │                        │
         ▼                     ▼                        ▼
 ┌────────────────┐   ┌─────────────────┐      ┌─────────────────┐
 │  Base Poisson   │   │  GAP Poisson    │      │  Calibrated ML   │
 │  (stable)       │   │  (advanced)     │      │ LightGBM/sklearn │
 └────────┬────────┘   └────────┬────────┘      └────────┬────────┘
          │                     │                         │
          └───────────┬─────────┴────────────┬────────────┘
                       ▼                      ▼
              ┌─────────────────┐    ┌──────────────────┐
              │  Final consensus  │───▶│ Certainty filter  │
              │ (weighted blend)  │    │ (blocks weak/     │
              └─────────────────┘    │ contradictory calls)│
                                      └─────────┬─────────┘
                                                ▼
                                  Final prediction: Over / Under /
                                       DO NOT BET + confidence
```

**Why layers instead of a single model:** each layer covers the previous one's blind spot. The base Poisson never fails because it only needs historical goals. GAP (process, not outcome) and StatsBomb add precision when real shot/xG data exists, but they're **never forced** if that data is missing or imputed. ML adds non-linearity, but it's calibrated to not contradict Poisson without reason. The consensus stage prevents a single "odd" model from deciding the match on its own, and the certainty filter is what actually protects against overconfidence: if the signals disagree, the system **stays quiet** instead of taking a risk.

## 3. Pipeline components (file map)

| Stage | File | What it does |
|---|---|---|
| Data | `data_sources.py`, `fetch_fixtures.py`, `bot_fixtures_patch.py` | Fetches matches/fixtures (including Telegram integrations) |
| Dataset build | `build_dataset.py` | Builds the historical training dataset (with or without StatsBomb, `--no-statsbomb` flag) |
| Features | `features.py` | ELO, recent goal form, recent pace, rest days, tournament stage |
| Process rating | `gap_ratings.py` | GAP ratings (real shots/xG with fallback to goals) |
| Probabilistic models | `poisson_models.py` | Base Poisson + GAP Poisson |
| ML model | `ml_model.py` | Calibrated LightGBM, with a scikit-learn fallback if LightGBM isn't available |
| Final blend | `consensus.py` | Combines base Poisson + GAP Poisson + ML into a single probability |
| Risk control | `filter_certainty.py` | Blocks the bet (`NO APOSTAR`) if the signal is weak or contradictory |
| Training | `train.py` | Trains GAP, Poisson, and ML on the built dataset |
| Prediction | `predict.py` | Predicts a single match or a CSV of matches |
| Validation | `backtest.py` | Quick backtest of the certainty filter against historical data |
| Orchestrator | `run_system.py` | Runs the full pipeline end to end |
| Configuration | `config.py` | All system parameters in one place |
| Utilities | `utils.py` | Shared helper functions |
| Telegram bot | `telegram_bot_free.py`, `telegram_fixtures_manager.py` | User-facing Telegram interface: predictions and upcoming fixtures |
| Reference data | `partidos_mundial.csv`, `partidos_ejemplo.csv`, `telegram_fixtures_worldcup.csv`, `pm.csv` | Match datasets used by the system |

## 4. Installation

```bat
cd /d D:\Mundial\over25_ultra_free
pip install -r requirements.txt
```

LightGBM is optional but recommended (if it fails to install, the system automatically falls back to scikit-learn):

```bat
pip install lightgbm
```

## 5. Usage

**1. Build the historical dataset:**
```bat
python build_dataset.py
```
To skip the StatsBomb integration (faster, less precise):
```bat
python build_dataset.py --no-statsbomb
```

**2. Train the models:**
```bat
python train.py
```

**3. Predict a single match:**
```bat
python predict.py "South Africa" "Canada" --sede "Los Angeles" --fase "Dieciseisavos" --fecha "2026-06-28" --dias-local 3 --dias-visitante 3
```

**4. Predict a CSV of matches:**
```bat
python predict.py --input partidos_ejemplo.csv --output outputs\predicciones_ultra.csv
```

**5. Run the full pipeline:**
```bat
python run_system.py --input partidos_ejemplo.csv
```

## 6. Expected output

For each match, the system shows:

- Base Poisson Over probability
- GAP Poisson Over probability
- Calibrated ML Over probability
- Final Over/Under probability (consensus)
- Model confidence
- Betting confidence
- Any blocks applied by the certainty filter

If the output is `NO APOSTAR` (DO NOT BET), **that's not a failure** — it means the system doesn't detect a sufficient edge to recommend a position.

## 7. Deliberate design decisions

These are constraints the system enforces on purpose, to avoid the typical failure modes of this kind of project:

- Doesn't treat imputed (estimated) data as if it were real data.
- Doesn't let LightGBM contradict Poisson without control.
- Doesn't apply isotonic calibration when data is scarce (it overfits).
- Doesn't force variances to zero artificially.
- Doesn't show 100% confidence unless calibration genuinely justifies it, and even then the value is clipped between 0.1% and 99.9%.

## 8. Limitations

- No model is 100% reliable for sports betting; this system doesn't claim to be.
- Prediction quality depends on the availability of StatsBomb data (real shots/xG); without it, the system degrades in a controlled way to the goals-based fallback, but with lower precision.
- Designed and tuned for international tournaments like the World Cup; performance on domestic leagues hasn't been validated in the same way.

## 9. Repository structure

```
data/                                  # Raw/processed data
fotos/                                 # Visual assets
models/                                # Serialized trained models
outputs/                               # Generated predictions
over25_ultra_free_project_FIX2_src/    # Source code (working version)
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

## 10. Roadmap / status

Actively developed project, built iteratively (AI-assisted plus manual adjustments). See `Memoria_tecnica_bot_prediccion_over25_estilo...` for the full technical breakdown of the build process.
