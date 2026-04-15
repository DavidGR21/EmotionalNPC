"""
Orquestador del pipeline difuso completo.

Pipeline estandar:
1) Fuzzificacion
2) Inferencia
3) Agregacion
4) Desfuzzificacion
"""

from fuzzy.fuzzificacion import fuzzify_arousal, fuzzify_valence
from fuzzy.inferencia import evaluate_rule_activations
from fuzzy.agregacion import aggregate_emotions, map_emotions_to_actions, normalize_emotions, EMOTION_TO_ACTION
from fuzzy.desfuzzificacion import choose_emotion_max


def process_fuzzy_logic(emotion, genome: dict, include_action_scores: bool = False):
    """
    Ejecuta el sistema difuso sobre el estado emocional actual.

    Retorna:
    - decision: accion mapeada desde la emocion dominante.
    - dominant_emotion: emocion difusa con mayor activacion.
    - top_emotions: top 2 emociones para telemetria/hud.
    - action_scores/emotion_scores (opcional): distribuciones completas.
    """
    af = fuzzify_arousal(emotion.Arousal, genome)
    vf = fuzzify_valence(emotion.Valence, genome)

    rules = evaluate_rule_activations(af, vf)
    emotion_scores = normalize_emotions(aggregate_emotions(rules))
    dominant_emotion = choose_emotion_max(emotion_scores)

    if dominant_emotion is None:
        decision = None
    else:
        decision = EMOTION_TO_ACTION[dominant_emotion]

    action_scores = map_emotions_to_actions(emotion_scores)

    sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
    top_emotions = {k: round(v, 4) for k, v in sorted_emotions[:2]}

    if include_action_scores:
        return decision, dominant_emotion, top_emotions, action_scores, emotion_scores

    return decision, dominant_emotion, top_emotions
