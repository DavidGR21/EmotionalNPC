"""
Paso 4 del sistema difuso: Desfuzzificacion.

Selecciona una emocion dominante a partir del conjunto agregado de emociones.
"""


def choose_emotion_max(emotions):
    """
    Retorna la emocion con mayor activacion (argmax).

    Si no hay emociones, retorna None.
    """
    if not emotions:
        return None
    return max(emotions, key=emotions.get)


def choose_action_max(actions):
    """Alias utilitario para argmax generico sobre diccionarios de score."""
    if not actions:
        return None
    return max(actions, key=actions.get)


# Alias de compatibilidad con nomenclatura previa.
choose_action = choose_action_max
