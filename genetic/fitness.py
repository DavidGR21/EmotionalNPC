"""
Fachada de fitness para EmotionalNPC.

Este modulo mantiene la API publica esperada por train_ga.py:
    get_fitness_evaluator(personality_type)

La implementacion interna fue separada por responsabilidades:
- fitness_constants.py: constantes y pesos globales.
- fitness_regularization.py: regularizacion difusa y annealing.
- fitness_components.py: componentes base (supervivencia y consistencia).
- fitness_profiles.py: estrategias de personalidad (todas en un archivo).
- fitness_evaluator.py: orquestacion de simulacion y score final.
"""

from genetic.fitness_evaluator import evaluate_timeline
from genetic.fitness_profiles import get_personality_scorer


def get_fitness_evaluator(personality_type="calmada"):
    """
    Factory que retorna la funcion de fitness para la personalidad solicitada.

    Firma retornada:
        evaluator(genome, generation=1, total_generations=1) -> float

        Justificacion de diseno:
        - Mantiene estable el punto de integracion usado por train_ga.py.
        - Permite cambiar internamente la logica de fitness sin romper llamadas
            externas ni scripts de entrenamiento existentes.
    """
    scoring_logic = get_personality_scorer(personality_type)

    return lambda genome, generation=1, total_generations=1: evaluate_timeline(
        genome=genome,
        scoring_logic=scoring_logic,
        generation=generation,
        total_generations=total_generations,
    )
