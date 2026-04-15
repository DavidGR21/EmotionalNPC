"""
Funciones de regularizacion estructural del sistema difuso.

Objetivo:
- Evitar soluciones degeneradas que "ganan" fitness por artefactos geometricos
    de la particion difusa, en vez de por comportamiento robusto.

Estrategia:
1) Penalizar centros demasiado cercanos (colapso semantico).
2) Penalizar sigma fuera de una zona funcional recomendada.
3) Aplicar annealing para reducir esta presion al final del entrenamiento.
"""

from genetic.fitness_constants import (
    ANNEAL_END,
    ANNEAL_START,
    GAP_PENALTY_FACTOR,
    MIN_CENTER_GAP,
    SIGMA_HIGH,
    SIGMA_HIGH_PENALTY,
    SIGMA_LOW,
    SIGMA_LOW_PENALTY,
)


def fuzzy_regularization(genome):
    """
    Penaliza configuraciones difusas degeneradas para estabilizar el aprendizaje.

        Formulacion aproximada:
                pen_gap = sum(max(0, MIN_CENTER_GAP - gap_i) * GAP_PENALTY_FACTOR)
                pen_sigma =
                        max(0, SIGMA_LOW - sigma) * SIGMA_LOW_PENALTY +
                        max(0, sigma - SIGMA_HIGH) * SIGMA_HIGH_PENALTY
                penalty = pen_gap + pen_sigma

        Justificacion de valores:
        - MIN_CENTER_GAP=0.08: mantiene separacion minima interpretable entre etiquetas.
        - GAP_PENALTY_FACTOR=40: castiga colapso sin anular exploracion por completo.
        - SIGMA_LOW=0.07 y SIGMA_HIGH=0.30: zona practica de compromiso entre
            sensibilidad y suavidad.
        - Penalizacion asimetrica de sigma (60 abajo, 40 arriba): sigma muy pequeno
            tiende a producir decisiones demasiado crispadas e inestables.
    """
    penalty = 0.0

    centro_arousal_bajo = genome.get("estimulacion_optima_baja", 0.1)
    centro_arousal_medio = genome.get("estimulacion_optima_media", 0.5)
    centro_arousal_alto = genome.get("estimulacion_optima_alta", 0.9)
    umbral_bienestar_negativo = genome.get("umbral_bienestar_negativo", 0.1)
    umbral_bienestar_neutral = genome.get("umbral_bienestar_neutral", 0.5)
    umbral_bienestar_positivo = genome.get("umbral_bienestar_positivo", 0.9)

    a_gap_1 = centro_arousal_medio - centro_arousal_bajo
    a_gap_2 = centro_arousal_alto - centro_arousal_medio
    v_gap_1 = umbral_bienestar_neutral - umbral_bienestar_negativo
    v_gap_2 = umbral_bienestar_positivo - umbral_bienestar_neutral

    for gap in [a_gap_1, a_gap_2, v_gap_1, v_gap_2]:
        if gap < MIN_CENTER_GAP:
            penalty += (MIN_CENTER_GAP - gap) * GAP_PENALTY_FACTOR

    sigma = genome.get("tolerancia_estimulo", 0.15)
    if sigma < SIGMA_LOW:
        penalty += (SIGMA_LOW - sigma) * SIGMA_LOW_PENALTY
    if sigma > SIGMA_HIGH:
        penalty += (sigma - SIGMA_HIGH) * SIGMA_HIGH_PENALTY

    return penalty


def annealed_regularization(genome, generation, total_generations):
    """
    Regularizacion adaptativa: fuerte al inicio, suave al final.

    Formula:
        progress = (gen - 1) / (G - 1)
        weight = ANNEAL_START - (ANNEAL_START - ANNEAL_END) * progress
        reg = fuzzy_regularization(genome) * weight

    Intuicion:
    - Generaciones tempranas: prevenir regiones patológicas del espacio de busqueda.
    - Generaciones finales: permitir refinamiento de desempeno con menos sesgo.
    """
    raw_penalty = fuzzy_regularization(genome)
    total = max(1, total_generations)
    progress = (max(1, generation) - 1) / max(1, total - 1)
    weight = ANNEAL_START - (ANNEAL_START - ANNEAL_END) * progress
    return raw_penalty * weight
