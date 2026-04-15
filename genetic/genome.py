import random

"""
Representacion y operadores geneticos para el ADN de personalidad.

Resumen de decisiones:
- Genoma continuo: habilita busqueda fina sobre rasgos psicologicos.
- DNA_BOUNDS: define rangos biologicamente/semanticamente validos.
- Crossover BLX-alpha (alpha=0.15): mezcla heredable con ligera exploracion.
- Mutacion gaussiana (rate=0.1, escala=0.12*rango): variacion estable.
- Reparacion difusa: asegura orden de centros para preservar interpretabilidad.
"""

# ==========================================
# MOTOR BIOLÓGICO: ESTRUCTURA DEL GENOMA (ADN)
# ==========================================

# DNA_BOUNDS define las "reglas de la física" de nuestro NPC.
# Es un diccionario que le indica al algoritmo genético cuáles son los
# valores mínimos y máximos que puede tomar cualquier parámetro.
# Evita que el algoritmo cree NPCs que tengan caracteisticas negativos (lo cual
# rompería la lógica difusa) o valores absurdamente grandes.
DNA_BOUNDS = {
    # 'w' son Percepción (qué tanto le importa X estímulo).
    # Rango de 0.0 (lo ignora completamente) a 1.0 (le afecta al máximo).
    "recepcion_sonoro": (0.0, 1.0),
    "recepcion_amenaza": (0.0, 1.0),
    "recepcion_nivel_luz": (0.0, 1.0),
    "recepcion_estres": (0.0, 1.0),
    "comfort_luz": (0.0, 1.0),
    "vulnerabilidad_peligro": (0.0, 1.0),
    "misofonia": (0.0, 1.0),
    "vulnerabilidad_estres": (0.0, 1.0),
    
    # 'k' son las Constantes Sensoriales de Estrés.
    # Evitamos que lleguen a 0 puro para prevenir errores de división (div_by_zero) en las fórmulas.
    "sensibilidad_amenaza": (0.05, 1.0),
    "sensibilidad_auditiva": (0.05, 1.0),
    "sensibilidad_oscuridad": (0.05, 1.0),
    
    # 'd' son las Tasas de Decaimiento (Inercia Emocional).
    # Determinan qué tan rápido cambia la emoción. Rango [0.05 a 1.0].
    # Cerca de 0.05: Emoción muuuuy lenta. Cambia lentamente y dura mucho (ej. un trauma).
    # Cerca de 1.0: Cambia instantáneamente. Se enoja/calma en un milisegundo.
    "volatilidad": (0.05, 1.0),
    "estabilidad_emocional": (0.05, 1.0),
    "resiliencia_estres": (0.05, 1.0),

    # Parámetros difusos optimizables (fuzzifier)
    # Centros para Arousal: low < medium < high
    "estimulacion_optima_baja": (0.0, 1.0),
    "estimulacion_optima_media": (0.0, 1.0),
    "estimulacion_optima_alta": (0.0, 1.0),
    # Centros para Valence: negative < neutral < positive
    "umbral_bienestar_negativo": (0.0, 1.0),
    "umbral_bienestar_neutral": (0.0, 1.0),
    "umbral_bienestar_positivo": (0.0, 1.0),
    # Sigmas independientes (por ahora el sistema sigue usando solo tolerancia_estimulo)
    "tolerancia_estimulo": (0.05, 0.35),
    "tolerancia_incomodidad": (0.05, 0.35),
}


def _repair_fuzzy_genes(genome):
    """
    Corrige genes difusos para que cumplan restricciones de orden y rangos.

        Justificacion:
        - Sin orden de centros (bajo <= medio <= alto), las etiquetas linguisticas
            dejan de representar particiones coherentes del dominio.
        - El clipping final evita valores fuera de DNA_BOUNDS tras cruce/mutacion.
    """
    repaired = genome.copy()

    a_centers = sorted([
        repaired["estimulacion_optima_baja"],
        repaired["estimulacion_optima_media"],
        repaired["estimulacion_optima_alta"],
    ])
    (
        repaired["estimulacion_optima_baja"],
        repaired["estimulacion_optima_media"],
        repaired["estimulacion_optima_alta"],
    ) = a_centers

    v_centers = sorted([
        repaired["umbral_bienestar_negativo"],
        repaired["umbral_bienestar_neutral"],
        repaired["umbral_bienestar_positivo"],
    ])
    (
        repaired["umbral_bienestar_negativo"],
        repaired["umbral_bienestar_neutral"],
        repaired["umbral_bienestar_positivo"],
    ) = v_centers

    for key, (min_val, max_val) in DNA_BOUNDS.items():
        repaired[key] = max(min_val, min(max_val, repaired[key]))

    return repaired


def generate_random_genome():
    """
    Paso 1 del Algoritmo Genético: Creación.
    Genera un individuo (NPC) desde cero, asignando a todos sus
    genes un valor completamente al azar dentro de sus límites.
    """
    genome = {}
    for key, (min_val, max_val) in DNA_BOUNDS.items():
        # random.uniform genera flotantes aleatorios entre el mínimo y máximo
        genome[key] = random.uniform(min_val, max_val)
    return _repair_fuzzy_genes(genome)


def crossover(parent1, parent2):
    """
    Paso 3A del Algoritmo Genético: Reproducción Sexual (Cruce).
    Toma los 'params' de dos NPCs que hayan sacado un buen puntaje (padres)
    y crea un nuevo hijo mezclando sus genes.
    
    Para parámetros continuos usamos BLX-alpha (Blend Crossover),
    que mezcla genes y permite explorar un poco fuera del rango entre padres.

    Valor usado: alpha=0.15.
    Razon: mantiene sesgo hacia rasgos parentales, pero habilita saltos cortos
    para escapar de zonas planas del espacio de busqueda.
    """
    child = {}
    alpha = 0.15
    for key in DNA_BOUNDS.keys():
        p1 = parent1[key]
        p2 = parent2[key]
        lo = min(p1, p2)
        hi = max(p1, p2)
        span = hi - lo
        child_val = random.uniform(lo - alpha * span, hi + alpha * span)

        min_val, max_val = DNA_BOUNDS[key]
        child[key] = max(min_val, min(max_val, child_val))

    return _repair_fuzzy_genes(child)


def mutate(genome, mutation_rate=0.1):
    """
    Paso 3B del Algoritmo Genético: Mutación.
    Previene el decaimiento cromosómico asegurando que siempre haya posibilidad
    de descubrir una nueva combinación (Escapar de Óptimos Locales).
    
    Por cada gen en el hijo resultante, tiramos un dado. Si el dado es
    menor a nuestra 'mutation_rate' (ej: 0.1 = 10% de probabilidad),
    entonces alteramos ese gen ligeramente sumandole o restandole ruido aleatorio.

    Valor usado para ruido: sigma = 0.12 * (max_val - min_val).
    Razon: una escala relativa al rango evita pasos excesivos en genes sensibles
    (p. ej. sigmas difusos) y pasos insignificantes en genes amplios.
    """
    mutated = genome.copy()
    
    for key, (min_val, max_val) in DNA_BOUNDS.items():
        if random.random() < mutation_rate:
            # Escalamos la mutación según el rango del gen para no desestabilizar
            # genes sensibles (por ejemplo sigma) ni congelar genes amplios.
            scale = (max_val - min_val) * 0.12
            noise = random.gauss(0, scale)
            mutated[key] += noise
            
            # Tras mutar, debemos pinzar (clamp) el valor para asegurar 
            # de que el ruido no lo sacó de sus límites legales impuestos en DNA_BOUNDS.
            mutated[key] = max(min_val, min(max_val, mutated[key]))

    return _repair_fuzzy_genes(mutated)
