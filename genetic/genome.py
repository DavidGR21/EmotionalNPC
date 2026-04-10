import random

# Definimos los límites para cada componente del `params` dict.
# Los pesos suelen estar normalizados entre 0 y 1.
# Las constantes de velocidad (k1, k2, k3) y decaimiento o inercia (dA, dV, ds) también.
DNA_BOUNDS = {
    "w1": (0.0, 1.0), "w2": (0.0, 1.0), "w3": (0.0, 1.0), "w4": (0.0, 1.0),
    "w5": (0.0, 1.0), "w6": (0.0, 1.0), "w7": (0.0, 1.0), "w8": (0.0, 1.0),
    "k1": (0.05, 1.0), "k2": (0.05, 1.0), "k3": (0.05, 1.0),
    "dA": (0.05, 1.0), "dV": (0.05, 1.0), "ds": (0.05, 1.0),
}


def generate_random_genome():
    """Genera un diccionario de params completamente aleatorio dentro de los bounds."""
    genome = {}
    for key, (min_val, max_val) in DNA_BOUNDS.items():
        genome[key] = random.uniform(min_val, max_val)
    return genome


def crossover(parent1, parent2):
    """Mezcla dos diccionarios params. Usamos cruce uniforme (mitad mamá, mitad papá)."""
    child = {}
    for key in DNA_BOUNDS.keys():
        if random.random() > 0.5:
            child[key] = parent1[key]
        else:
            child[key] = parent2[key]
    return child


def mutate(genome, mutation_rate=0.1):
    """Aplica mutaciones gaussianas aleatorias dependiendo de la tasa de mutación."""
    mutated = genome.copy()
    for key, (min_val, max_val) in DNA_BOUNDS.items():
        if random.random() < mutation_rate:
            # Agregamos ruido (ej: +/- 0.1 a 0.2)
            noise = random.gauss(0, 0.2)
            mutated[key] += noise
            # Aseguramos que se mantenga en los límites validos
            mutated[key] = max(min_val, min(max_val, mutated[key]))
    return mutated
