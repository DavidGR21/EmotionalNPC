import random

# ==========================================
# MOTOR BIOLÓGICO: ESTRUCTURA DEL GENOMA (ADN)
# ==========================================

# DNA_BOUNDS define las "reglas de la física" de nuestro NPC.
# Es un diccionario que le indica al algoritmo genético cuáles son los
# valores mínimos y máximos que puede tomar cualquier parámetro.
# Evita que el algoritmo cree NPCs que tengan pesos negativos (lo cual
# rompería la lógica difusa) o valores absurdamente grandes.
DNA_BOUNDS = {
    # 'w' son los Pesos de Percepción (qué tanto le importa X estímulo).
    # Rango de 0.0 (lo ignora completamente) a 1.0 (le afecta al máximo).
    "w1": (0.0, 1.0), "w2": (0.0, 1.0), "w3": (0.0, 1.0), "w4": (0.0, 1.0),
    "w5": (0.0, 1.0), "w6": (0.0, 1.0), "w7": (0.0, 1.0), "w8": (0.0, 1.0),
    
    # 'k' son las Constantes Sensoriales de Estrés.
    # Evitamos que lleguen a 0 puro para prevenir errores de división (div_by_zero) en las fórmulas.
    "k1": (0.05, 1.0), "k2": (0.05, 1.0), "k3": (0.05, 1.0),
    
    # 'd' son las Tasas de Decaimiento (Inercia Emocional).
    # Determinan qué tan rápido cambia la emoción. Rango [0.05 a 1.0].
    # Cerca de 0.05: Emoción muuuuy lenta. Cambia lentamente y dura mucho (ej. un trauma).
    # Cerca de 1.0: Cambia instantáneamente. Se enoja/calma en un milisegundo.
    "dA": (0.05, 1.0), "dV": (0.05, 1.0), "ds": (0.05, 1.0),
}


def generate_random_genome():
    """
    Paso 1 del Algoritmo Genético: Creación.
    Genera un individuo (NPC) desde cero, asignando a todos sus 
    14 parámetros un valor completamente al azar dentro de sus límites.
    """
    genome = {}
    for key, (min_val, max_val) in DNA_BOUNDS.items():
        # random.uniform genera flotantes aleatorios entre el mínimo y máximo
        genome[key] = random.uniform(min_val, max_val)
    return genome


def crossover(parent1, parent2):
    """
    Paso 3A del Algoritmo Genético: Reproducción Sexual (Cruce).
    Toma los 'params' de dos NPCs que hayan sacado un buen puntaje (padres)
    y crea un nuevo hijo mezclando sus genes.
    
    Se utiliza "Cruce Uniforme": Cada gen tiene un 50% de probabilidad
    de ser heredado de la mamá y 50% del papá.
    """
    child = {}
    for key in DNA_BOUNDS.keys():
        if random.random() > 0.5:
            child[key] = parent1[key] # Hereda gen del padre 1
        else:
            child[key] = parent2[key] # Hereda gen del padre 2
    return child


def mutate(genome, mutation_rate=0.1):
    """
    Paso 3B del Algoritmo Genético: Mutación.
    Previene el decaimiento cromosómico asegurando que siempre haya posibilidad
    de descubrir una nueva combinación (Escapar de Óptimos Locales).
    
    Por cada gen en el hijo resultante, tiramos un dado. Si el dado es
    menor a nuestra 'mutation_rate' (ej: 0.1 = 10% de probabilidad), 
    entonces alteramos ese gen ligeramente sumándole o restándole ruido aleatorio.
    """
    mutated = genome.copy()
    
    for key, (min_val, max_val) in DNA_BOUNDS.items():
        if random.random() < mutation_rate:
            # random.gauss(media, desviacion_estandar) agrega alteraciones naturales.
            # En este caso oscilará cerca de +/- 0.2
            noise = random.gauss(0, 0.2)
            mutated[key] += noise
            
            # Tras mutar, debemos pinzar (clamp) el valor para asegurar 
            # de que el ruido no lo sacó de sus límites legales impuestos en DNA_BOUNDS.
            mutated[key] = max(min_val, min(max_val, mutated[key]))
            
    return mutated
