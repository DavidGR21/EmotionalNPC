"""
Interfaz publica del paquete genetic.

Exports:
- DNA_BOUNDS, generate_random_genome, crossover, mutate: operadores del AG.
- get_fitness_evaluator: factory de evaluador de aptitud por personalidad.

Objetivo:
Concentrar en un unico punto los simbolos que consume el pipeline de
entrenamiento para simplificar imports y mantener estabilidad de API.
"""

from .genome import DNA_BOUNDS, generate_random_genome, crossover, mutate
from .fitness import get_fitness_evaluator
