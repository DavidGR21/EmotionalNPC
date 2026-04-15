"""
Paso 3 del sistema difuso: Agregacion.

Combina reglas elementales por emocion usando max y opcionalmente normaliza
para obtener puntajes comparables en [0, 1].
"""

EMOTION_TO_ACTION = {
    "miedo": "flee",
    "tristeza": "idle",
    "felicidad": "explore",
    "calma": "dormir",
}


def aggregate_emotions(rule_activations):
    """
    Agrega activaciones de reglas en emociones finales.

    Operador usado: max (union difusa) para cada emocion.
    """
    return {
        "miedo": max(rule_activations["miedo_hn"], rule_activations["miedo_mn"]),
        "tristeza": rule_activations["tristeza_ln"],
        "felicidad": max(rule_activations["felicidad_hp"], rule_activations["felicidad_mp"]),
        "calma": max(rule_activations["calma_lu"], rule_activations["calma_lp"]),
    }


def normalize_emotions(emotions):
    """Normaliza scores de emociones para que la suma sea 1 si total > 0."""
    total = sum(emotions.values())
    if total == 0:
        return emotions
    return {k: v / total for k, v in emotions.items()}


def map_emotions_to_actions(emotion_scores):
    """Convierte puntajes emocionales en puntajes de accion usando mapeo 1-a-1."""
    action_scores = {}
    for emotion, score in emotion_scores.items():
        action = EMOTION_TO_ACTION[emotion]
        action_scores[action] = action_scores.get(action, 0.0) + score
    return action_scores


# Alias de compatibilidad con nomenclatura previa.
aggregate_actions = aggregate_emotions
normalize_actions = normalize_emotions
