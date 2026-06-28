from __future__ import annotations

import json
from functools import lru_cache
from typing import Dict, Any, List, Tuple

from config import SETTINGS, PATHS


@lru_cache(maxsize=1)
def _load_optimized_thresholds() -> Dict[str, Any]:
    """
    Lee los umbrales recomendados por backtest.py --apply-best.
    Si no existe el archivo, usa SETTINGS.

    FIX 3:
    - Aunque exista un JSON antiguo del FIX 2, si no trae filter_mode se usa
      por defecto el nuevo modo directional_consensus.
    - No borra ni ignora métricas útiles antiguas: max_model_discrepancy sigue
      disponible para comparar/backtest legacy, pero ya no bloquea por defecto.
    """
    if not SETTINGS.get("use_optimized_thresholds", True):
        return {}
    path = PATHS.get("recommended_thresholds")
    if not path or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _num(value, default: float) -> float:
    """Convierte valores de JSON/config a float de forma segura.

    FIX 3B:
    backtest.py puede guardar max_model_discrepancy=None cuando el modo
    recomendado es directional_consensus. float(None) rompía predict.py.
    Esta función conserva el valor por defecto si llega None, NaN o texto raro.
    """
    try:
        if value is None:
            return float(default)
        v = float(value)
        if v != v:  # NaN
            return float(default)
        return v
    except Exception:
        return float(default)


def _thresholds() -> Dict[str, Any]:
    opt = _load_optimized_thresholds()
    margin = _num(opt.get("directional_margin", SETTINGS.get("directional_margin", 0.05)), SETTINGS.get("directional_margin", 0.05))
    return {
        "confidence_threshold": _num(opt.get("confidence_threshold"), SETTINGS["confidence_threshold"]),
        "confidence_threshold_strict": _num(opt.get("confidence_threshold_strict"), SETTINGS["confidence_threshold_strict"]),
        # En directional_consensus puede venir None desde recommended_thresholds_ultra.json.
        # Lo dejamos caer al valor de config porque en ese modo solo se informa, no bloquea.
        "max_model_discrepancy": _num(opt.get("max_model_discrepancy"), SETTINGS["max_model_discrepancy"]),
        "directional_margin": margin,
        "under_confirm_ceiling": _num(opt.get("under_confirm_ceiling"), SETTINGS.get("under_confirm_ceiling", 0.50 - margin)),
        "over_confirm_floor": _num(opt.get("over_confirm_floor"), SETTINGS.get("over_confirm_floor", 0.50 + margin)),
        "filter_mode": opt.get("filter_mode", SETTINGS.get("filter_mode", "directional_consensus")) or SETTINGS.get("filter_mode", "directional_consensus"),
        "source": opt.get("source", "config.py"),
    }


def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        v = float(value)
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default


def _model_probs(consensus: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Probabilidades P(Over) de cada submodelo disponible."""
    pairs = []
    mapping = [
        ("poisson_base", "p_over_poisson_base"),
        ("poisson_gap", "p_over_poisson_gap"),
        ("ml", "p_over_ml"),
    ]
    for label, key in mapping:
        v = _safe_float(consensus.get(key), None)
        if v is not None:
            pairs.append((label, max(0.001, min(0.999, v))))
    return pairs


def _directional_consensus_check(pred_raw: str, consensus: Dict[str, Any], th: Dict[str, Any]) -> tuple[bool, str, str]:
    """
    Nuevo filtro FIX 3: consenso direccional.

    UNDER válido: ningún modelo puede estar en zona Over/duda.
      => todos los P(Over) deben quedar por debajo del techo UNDER.
    OVER válido: ningún modelo puede estar en zona Under/duda.
      => todos los P(Over) deben quedar por encima del suelo OVER.

    Esto evita que una discrepancia grande bloquee cuando todos los modelos
    realmente apuntan al mismo lado. Ejemplo: 28%, 41%, 44.9% => UNDER OK.
    Y bloquea contradicciones reales: 62%, 53%, 61% => OVER NO OK.
    """
    probs = _model_probs(consensus)
    if not probs:
        return False, "sin_modelos_para_consenso_direccional", "sin_modelos"

    under_ceiling = float(th["under_confirm_ceiling"])
    over_floor = float(th["over_confirm_floor"])

    if pred_raw.startswith("UNDER"):
        max_label, max_prob = max(probs, key=lambda x: x[1])
        # La usuaria pidió: si uno llega a 45% o más, ya no confirmar UNDER.
        # Usamos >=, pero si el valor real es 0.44978 y se imprime 45.0%, pasa.
        if max_prob >= under_ceiling:
            return (
                False,
                f"contradiccion_under({max_label}={max_prob:.3f}>={under_ceiling:.3f})",
                f"consenso_direccional_under_NO_ok(max={max_prob:.3f}, techo={under_ceiling:.3f})",
            )
        return (
            True,
            f"consenso_direccional_under_ok(max={max_prob:.3f}<{under_ceiling:.3f})",
            f"under_confirmado(max_model={max_prob:.3f})",
        )

    # OVER válido: todos los modelos deben estar al menos en 55% si el margen es 5%.
    min_label, min_prob = min(probs, key=lambda x: x[1])
    if min_prob < over_floor:
        return (
            False,
            f"contradiccion_over({min_label}={min_prob:.3f}<{over_floor:.3f})",
            f"consenso_direccional_over_NO_ok(min={min_prob:.3f}, suelo={over_floor:.3f})",
        )
    return (
        True,
        f"consenso_direccional_over_ok(min={min_prob:.3f}>={over_floor:.3f})",
        f"over_confirmado(min_model={min_prob:.3f})",
    )


def apply_filter(features: Dict[str, Any], consensus: Dict[str, Any], ml_status: str = "ok") -> Dict[str, Any]:
    p = float(consensus.get("p_over_final", 0.5))
    conf = float(consensus.get("confidence_model", 0.5))
    pred_raw = "OVER 2.5" if p >= 0.5 else "UNDER 2.5"
    fails = []
    checks = []

    th = _thresholds()

    min_matches = int(SETTINGS["min_team_matches"])
    lp = int(float(features.get("local_partidos_previos", 0)))
    vp = int(float(features.get("visitante_partidos_previos", 0)))
    if lp >= min_matches and vp >= min_matches:
        checks.append("datos_equipo_ok")
    else:
        fails.append(f"datos_insuficientes_equipo({lp}/{vp})")

    threshold = float(th["confidence_threshold"])
    if ml_status != "ok":
        checks.append(f"ml_no_usado:{ml_status}")
        threshold = max(threshold, float(th["confidence_threshold_strict"]))
    else:
        checks.append("ml_ok")

    # FIX 3: la discrepancia absoluta se mantiene como INFO. Por defecto ya no bloquea.
    discrepancy = float(consensus.get("discrepancy", 0.0))
    max_discrepancy = float(th["max_model_discrepancy"])
    filter_mode = str(th.get("filter_mode", "directional_consensus"))

    if filter_mode == "legacy_discrepancy":
        if discrepancy <= max_discrepancy:
            checks.append(f"modelos_coherentes_legacy({discrepancy:.3f}<={max_discrepancy:.3f})")
        else:
            fails.append(f"discrepancia_modelos_alta_legacy({discrepancy:.3f}>{max_discrepancy:.3f})")
    elif filter_mode == "hybrid_both":
        ok_dir, fail_msg, check_msg = _directional_consensus_check(pred_raw, consensus, th)
        if ok_dir:
            checks.append(check_msg)
        else:
            fails.append(fail_msg)
        if discrepancy <= max_discrepancy:
            checks.append(f"discrepancia_legacy_ok({discrepancy:.3f}<={max_discrepancy:.3f})")
        else:
            fails.append(f"discrepancia_modelos_alta_legacy({discrepancy:.3f}>{max_discrepancy:.3f})")
    else:
        ok_dir, fail_msg, check_msg = _directional_consensus_check(pred_raw, consensus, th)
        if ok_dir:
            checks.append(check_msg)
        else:
            fails.append(fail_msg)
        checks.append(f"discrepancia_info({discrepancy:.3f};no_bloquea_en_modo_direccional)")

    votes_over = int(consensus.get("votes_over", 0))
    votes_under = int(consensus.get("votes_under", 0))
    if votes_over == votes_under and conf < float(th["confidence_threshold_strict"]):
        fails.append("votos_divididos_y_confianza_baja")
    else:
        checks.append("votos_utilizables")

    if conf >= threshold:
        checks.append(f"confianza_ok({conf:.3f}>={threshold:.3f})")
    else:
        fails.append(f"confianza_baja({conf:.3f}<{threshold:.3f})")

    vg_l = float(features.get("varianza_goles_local_8", 0.0) or 0.0)
    vg_v = float(features.get("varianza_goles_visit_8", 0.0) or 0.0)
    if max(vg_l, vg_v) <= 6.0:
        checks.append("varianza_goles_controlada")
    else:
        fails.append("varianza_goles_muy_alta")

    vs_l = float(features.get("varianza_shots_local_8", 0.0) or 0.0)
    vs_v = float(features.get("varianza_shots_visit_8", 0.0) or 0.0)
    stats_ok = int(features.get("statsbomb_available", 0)) == 1

    if stats_ok:
        if max(vs_l, vs_v) <= SETTINGS.get("max_shot_variance", 25.0):
            checks.append("varianza_shots_controlada")
        else:
            fails.append(f"varianza_shots_muy_alta({max(vs_l, vs_v):.1f})")
    else:
        checks.append("varianza_shots_ignorada(sin_statsbomb)")

    if fails:
        decision = "NO APOSTAR"
        bet_confidence = 0.0
    else:
        decision = pred_raw
        bet_confidence = conf

    return {
        "decision": decision,
        "observacion_no_apuesta": pred_raw,
        "p_over_final": p,
        "p_under_final": 1.0 - p,
        "confidence_model": conf,
        "confidence_bet": bet_confidence,
        "checks_ok": "|".join(checks),
        "bloqueos": "|".join(fails) if fails else "ninguno",
        "threshold_used": threshold,
        "max_discrepancy_used": max_discrepancy,
        "directional_margin_used": float(th["directional_margin"]),
        "under_confirm_ceiling_used": float(th["under_confirm_ceiling"]),
        "over_confirm_floor_used": float(th["over_confirm_floor"]),
        "filter_mode": filter_mode,
        "threshold_source": th["source"],
    }