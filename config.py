from __future__ import annotations

from pathlib import Path

# Directorios del sistema
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"

# Crear directorios si no existen
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# URLs de descarga de datos (ejemplos estructurados para StatsBomb y repositorios públicos)
URLS = {
    "international_results": "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
    "statsbomb_competitions": "https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json",
    "statsbomb_base": "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
}

# Paths internos de los archivos del ecosistema
PATHS = {
    "raw_results": RAW_DIR / "results.csv",
    "statsbomb_competitions": RAW_DIR / "competitions.json",
    "statsbomb_match_stats": RAW_DIR / "statsbomb_match_stats.csv",
    "dataset": RAW_DIR / "dataset_ultra_final.csv",
    "team_state": RAW_DIR / "team_state_ultra.csv",
    "artifact": OUTPUT_DIR / "model_artifact_ultra.pkl",
    "cv_results": OUTPUT_DIR / "cv_results_temporal.csv",
    "feature_importance": OUTPUT_DIR / "feature_importance.csv",
    "train_report": OUTPUT_DIR / "train_report.txt",
    "recommended_thresholds": OUTPUT_DIR / "recommended_thresholds_ultra.json",
    "build_report": OUTPUT_DIR / "build_report.txt"
}

# Configuración global del modelo y los hiperparámetros
SETTINGS = {
    # Filtros temporales y de volumen
    "min_year": 2010,
    "max_year_train": 2026,
    "train_min_year": 2014,
    "min_team_matches": 5,
    "min_rows_train_ml": 500,
    
    # Configuración de StatsBomb
    "use_statsbomb": True,
    "max_statsbomb_matches": 5000,
    
    # Configuración base de ELO
    "elo_default": 1500.0,
    "elo_home_advantage": 65.0,
    "elo_k_base": 40.0,
    
    # Ventanas de forma (Partidos previos a evaluar)
    "form_window": 8,
    "long_form_window": 25,
    
    # Promedios globales de referencia
    "world_avg_total_goals": 2.65,
    
    # Hiperparámetros del modelo GAP (Ataque/Defensa basados en tiros)
    "gap_init_attack": 10.0,
    "gap_init_defense": 10.0,
    "gap_min": 2.0,
    "gap_max": 35.0,
    "gap_corner_weight": 0.15,
    "gap_real_stats_weight": 1.0,
    "gap_fallback_weight": 0.25,
    "gap_goal_to_process": 4.5,
    "gap_phi_attack": 0.08,
    "gap_phi_defense": 0.06,
    
    # Límites para los modelos de Poisson
    "lambda_min": 0.15,
    "lambda_total_min": 0.60,
    "lambda_total_max": 5.50,
    
    # Ajustes geográficos y ambientales (Altitud)
    "altitude_threshold": 1200.0,
    "altitude_lambda_penalty": 0.07,
    
    # Configuración del modelo ML y validación
    "random_state": 42,
    "calibration_ratio": 0.20,
    
    # Filtros de Certeza y Gestión de Riesgo (Filtro por Varianza)
    "use_optimized_thresholds": False,
    "confidence_threshold": 0.164,
    "confidence_threshold_strict": 0.219,
    "max_model_discrepancy": 0.35,
    "directional_margin": 0.05,
    "filter_mode": "legacy_discrepancy",  # Modos: 'legacy_discrepancy', 'hybrid_both', 'directional_consensus'
    
    # NUEVA MEJORA: Umbral crítico para varianza de tiros en los últimos 8 partidos
    "max_shot_variance": 28.5,
    
    # NUEVA MEJORA: Pesos dinámicos del consenso de probabilidad
    # Caso A: Con estadísticas avanzadas reales disponibles (StatsBomb OK)
    "weight_poisson_base_with_stats": 0.15,
    "weight_poisson_gap_with_stats": 0.35,
    "weight_ml_with_stats": 0.50,
    
    # Caso B: Sin estadísticas avanzadas (Fallo de scraping o partido no cubierto)
    "weight_poisson_base_no_stats": 0.45,
    "weight_poisson_gap_no_stats": 0.40,
    "weight_ml_no_stats": 0.15
}

# Sedes y altitudes conocidas para cálculo de penalizaciones por oxígeno
VENUES = {
    "La Paz": {"altitude": 3625.0, "country": "Bolivia"},
    "Quito": {"altitude": 2850.0, "country": "Ecuador"},
    "Bogota": {"altitude": 2625.0, "country": "Colombia"},
    "Mexico City": {"altitude": 2240.0, "country": "Mexico"},
    "Denver": {"altitude": 1609.0, "country": "USA"},
    "Johannesburg": {"altitude": 1750.0, "country": "South Africa"}
}

# Fases del torneo y sus códigos/penalizaciones (fútbol de eliminación directa tiende a cerrarse)
PHASES = {
    "Grupo": {"codigo": 1, "lambda_penalty": 0.0},
    "Dieciseisavos": {"codigo": 2, "lambda_penalty": 0.02},
    "Octavos": {"codigo": 3, "lambda_penalty": 0.04},
    "Cuartos": {"codigo": 4, "lambda_penalty": 0.06},
    "Semis": {"codigo": 5, "lambda_penalty": 0.09},
    "Tercer_puesto": {"codigo": 6, "lambda_penalty": 0.01},
    "Final": {"codigo": 7, "lambda_penalty": 0.12}
}

# Pesos de importancia de torneos oficiales para el multiplicador de K en ELO
TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 1.0,
    "UEFA Euro": 0.90,
    "Copa America": 0.85,
    "Copa América": 0.85,
    "African Cup of Nations": 0.80,
    "AFC Asian Cup": 0.75,
    "CONCACAF Gold Cup": 0.75,
    "FIFA World Cup qualification": 0.65,
    "UEFA Euro qualification": 0.60,
    "Friendly": 0.35
}

# Fragmentos válidos para admitir torneos en el pipeline
VALID_TOURNAMENT_FRAGMENTS = [
    "World Cup", "Euro", "Copa", "America", "América", "African", "Asian", "Gold Cup", "Nations League", "Friendly"
]

# Fragmentos explícitos para ignorar torneos espurios o de categorías inferiores
EXCLUDED_TOURNAMENT_FRAGMENTS = [
    "U-21", "U-20", "U-19", "U-17", "Women", "Femenino", "Indoor", "Military"
]

# Mapeo de normalización de nombres de equipos de fútbol para consistencia del dataset
TEAM_ALIASES = {
    "USA": "United States",
    "EE.UU.": "United States",
    "Estados Unidos": "United States",
    "South Korea": "Korea Republic",
    "Corea del Sur": "Korea Republic",
    "Iran": "IR Iran",
    "Irán": "IR Iran",
    "North Korea": "Korea DPR",
    "Corea del Norte": "Korea DPR",
    "Ivory Coast": "Côte d'Ivoire",
    "Costa de Marfil": "Côte d'Ivoire",
    "Republic of Ireland": "Ireland",
    "Irlanda": "Ireland",
    "Czechia": "Czech Republic",
    "República Checa": "Czech Republic",
    "Marruecos": "Morocco",
    "España": "Spain",
    "Alemania": "Germany",
    "Francia": "France",
    "Inglaterra": "England",
    "Italia": "Italy",
    "Brasil": "Brazil",
    "Argentina": "Argentina"
}