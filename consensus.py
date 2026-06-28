from __future__ import annotations

from typing import Dict, Any
from config import SETTINGS


def combine_probabilities(features: Dict[str, Any], p_ml: float | None) -> Dict[str, Any]:
    """
    Combina las probabilidades de los modelos de Poisson (Base y GAP) con el 
    modelo de Machine Learning, aplicando un esquema de pesos dinámicos según 
    la disponibilidad de estadísticas avanzadas (StatsBomb).
    """
    p_base = float(features.get("p_over_poisson_base", 0.5))
    p_gap = float(features.get("p_over_poisson_gap", 0.5))
    stats_ok = int(features.get("statsbomb_available", 0)) == 1

    # Ajuste dinámico de pesos según calidad del scraping (con o sin StatsBomb)
    if stats_ok:
        w_base = float(SETTINGS.get("weight_poisson_base_with_stats", 0.20))
        w_gap = float(SETTINGS.get("weight_poisson_gap_with_stats", 0.30))
        w_ml = float(SETTINGS.get("weight_ml_with_stats", 0.50))
    else:
        w_base = float(SETTINGS.get("weight_poisson_base_no_stats", 0.40))
        w_gap = float(SETTINGS.get("weight_poisson_gap_no_stats", 0.40))
        w_ml = float(SETTINGS.get("weight_ml_no_stats", 0.20))

    # Salvaguarda si el modelo ML no está entrenado o devuelve un valor nulo
    if p_ml is None:
        total_w = w_base + w_gap
        if total_w > 0:
            w_base = w_base / total_w
            w_gap = w_gap / total_w
        else:
            w_base, w_gap = 0.50, 0.50
        w_ml = 0.0
        p_ml_val = 0.5
    else:
        p_ml_val = p_ml

    # Cálculo de la probabilidad final ponderada
    p_final = (p_base * w_base) + (p_gap * w_gap) + (p_ml_val * w_ml)

    # Aislar las probabilidades de los modelos con peso activo para métricas de consistencia
    active_probs = []
    if w_base > 0: active_probs.append(p_base)
    if w_gap > 0: active_probs.append(p_gap)
    if w_ml > 0: active_probs.append(p_ml_val)

    # Cálculo de la discrepancia real entre expertos activos
    discrepancy = max(active_probs) - min(active_probs) if active_probs else 0.0

    # Construcción del tracking de modelos utilizados
    models_used_list = []
    if w_base > 0: models_used_list.append("base")
    if w_gap > 0: models_used_list.append("gap")
    if w_ml > 0: models_used_list.append("ml")

    return {
        "p_over_final": p_final,
        "p_under_final": 1.0 - p_final,
        "p_over_poisson_base": p_base,
        "p_over_poisson_gap": p_gap,
        "p_over_ml": p_ml,
        "discrepancy": discrepancy,
        "confidence_model": abs(p_final - 0.5) * 2.0,
        "models_used": ",".join(models_used_list),
        "votes_over": sum(1 for p in active_probs if p >= 0.5),
        "votes_under": sum(1 for p in active_probs if p < 0.5),
    }