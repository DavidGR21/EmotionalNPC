"""
Orquestacion de la simulacion y calculo del score final de fitness.

Responsabilidad unica del modulo:
- Ejecutar la simulacion temporal de un genoma.
- Acumular los componentes (survival, consistency, personality).
- Normalizar, ponderar y aplicar regularizacion annealed.

Formula final:
    score = 0.35*S_hat + 0.25*C_hat + 0.40*P_hat - reg/100
"""

from models.emotion import Emotion
from fuzzy.sistema_difuso import process_fuzzy_logic
from cli_test import build_environment_timeline
from genetic.fitness_components import (
    component_consistency,
    component_survival,
    normalize_component_totals,
)
from genetic.fitness_constants import COMPONENT_WEIGHTS
from genetic.fitness_regularization import annealed_regularization


def evaluate_timeline(genome, scoring_logic, generation=1, total_generations=1):
    """
    Ejecuta una simulacion completa y retorna el fitness escalar final.

    Pipeline por tick:
    1) emotion.update(env, genome)
    2) process_fuzzy_logic(...) -> action + conf
    3) acumular S, C y P
    4) normalizar, ponderar y restar regularizacion

    Parametros:
    - genome: diccionario con genes del individuo.
    - scoring_logic: estrategia de personalidad (inyectada por factory).
    - generation/total_generations: necesarios para annealing de regularizacion.

    Retorno:
    - float: fitness final del individuo para esta evaluacion.
    """
    emotion = Emotion()
    timeline = build_environment_timeline()
    totals = {
        "survival": 0.0,
        "consistency": 0.0,
        "personality": 0.0,
    }
    prev_conf = None

    for phase, env in timeline:
        emotion.update(env, genome)
        decision, _, _, action_scores, _ = process_fuzzy_logic(
            emotion,
            genome,
            include_action_scores=True,
        )

        totals["survival"] += component_survival(env, action_scores)
        totals["consistency"] += component_consistency(env, emotion, action_scores, prev_conf)
        totals["personality"] += scoring_logic(phase, env, emotion, decision, action_scores)
        prev_conf = action_scores

    tick_count = max(1, len(timeline))
    normalized = normalize_component_totals(totals, tick_count)

    score = 0.0
    for name, weight in COMPONENT_WEIGHTS.items():
        score += weight * normalized[name]

    reg_penalty = annealed_regularization(genome, generation, total_generations)
    score -= reg_penalty / 100.0
    return score
