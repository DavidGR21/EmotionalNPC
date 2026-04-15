"""
Paso 2 del sistema difuso: Inferencia.

Evalua la base de reglas difusas aplicando operador AND=min entre etiquetas
de Arousal y Valence.
"""


def evaluate_rule_activations(af, vf):
    """
    Calcula el grado de activacion de cada regla elemental.

    Entradas:
    - af: membresias de arousal (low/medium/high).
    - vf: membresias de valence (negative/neutral/positive).

    Reglas organizadas por emociones del circumplejo de Russell:
    - miedo: alto arousal + valence negativa.
    - tristeza: bajo arousal + valence negativa.
    - felicidad: arousal medio/alto + valence positiva.
    - calma: bajo arousal + valence neutral/positiva.
    """
    return {
        "miedo_hn": min(af["high"], vf["negative"]),
        "miedo_mn": min(af["medium"], vf["negative"]),
        "tristeza_ln": min(af["low"], vf["negative"]),
        "felicidad_hp": min(af["high"], vf["positive"]),
        "felicidad_mp": min(af["medium"], vf["positive"]),
        "calma_lu": min(af["low"], vf["neutral"]),
        "calma_lp": min(af["low"], vf["positive"]),
    }


# Alias explicito para mantener semantica previa donde se esperaba "evaluate_rules".
evaluate_rules = evaluate_rule_activations
