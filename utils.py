from __future__ import annotations

import re
import unicodedata
from typing import Any

from config import TEAM_ALIASES, PHASES, TOURNAMENT_WEIGHTS, VALID_TOURNAMENT_FRAGMENTS, EXCLUDED_TOURNAMENT_FRAGMENTS


def strip_accents(text: str) -> str:
    text = str(text or "")
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def normalize_team(name: Any) -> str:
    name = normalize_spaces(name)
    if not name:
        return ""
    if name in TEAM_ALIASES:
        return TEAM_ALIASES[name]
    no_acc = strip_accents(name)
    if no_acc in TEAM_ALIASES:
        return TEAM_ALIASES[no_acc]
    return name


def phase_key(value: Any) -> str:
    value = normalize_spaces(value)
    aliases = {
        "round of 32": "Dieciseisavos", "round of32": "Dieciseisavos", "r32": "Dieciseisavos",
        "dieciseisavos de final": "Dieciseisavos", "dieciseisavos": "Dieciseisavos",
        "round of 16": "Octavos", "r16": "Octavos", "octavos": "Octavos", "octavos de final": "Octavos",
        "quarterfinal": "Cuartos", "quarterfinals": "Cuartos", "cuartos": "Cuartos",
        "semifinal": "Semis", "semi-final": "Semis", "semis": "Semis",
        "final": "Final", "3rd place": "Tercer_puesto", "third": "Tercer_puesto",
        "tercer puesto": "Tercer_puesto", "grupo": "Grupo", "group": "Grupo",
    }
    key = strip_accents(value).lower()
    return aliases.get(key, value if value in PHASES else "Grupo")


def phase_code(value: Any) -> int:
    return int(PHASES[phase_key(value)]["codigo"])


def phase_penalty(value: Any) -> float:
    return float(PHASES[phase_key(value)]["lambda_penalty"])


def tournament_weight(tournament: Any) -> float:
    t = normalize_spaces(tournament)
    if t in TOURNAMENT_WEIGHTS:
        return float(TOURNAMENT_WEIGHTS[t])
    lower = strip_accents(t).lower()
    if "qualification" in lower or "qualifier" in lower:
        return 0.65
    if "friendly" in lower or "friend" in lower:
        return 0.35
    if "world cup" in lower:
        return 1.0
    if "euro" in lower:
        return 0.90
    if "copa" in lower:
        return 0.85
    return 0.55


def valid_tournament(tournament: Any) -> bool:
    t = strip_accents(str(tournament or "")).lower()
    if any(x in t for x in EXCLUDED_TOURNAMENT_FRAGMENTS):
        return False
    if any(strip_accents(x).lower() in t for x in VALID_TOURNAMENT_FRAGMENTS):
        return True
    return False


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        if v != v:
            return default
        return v
    except Exception:
        return default
