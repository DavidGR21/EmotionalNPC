"""
Componentes base del fitness (supervivencia y consistencia).

Este modulo define subobjetivos generales, independientes del perfil de
personalidad. Su salida luego se combina con el componente de personalidad.
"""

from genetic.fitness_constants import (
    ACTIONS,
    MAX_ABS_COMPONENT_PER_TICK,
    MEDIUM_IMPACT,
    SOFT_IMPACT,
    THREAT_HIGH,
    THREAT_LOW,
)


def clip(value, limit):
    """
    Recorta un valor al intervalo [-limit, +limit].

    Se utiliza para controlar outliers por tick y mantener la escala estable.
    """
    return max(-limit, min(limit, value))


def normalize_component_totals(totals, tick_count):
    """
    Normaliza cada componente por tick y por su maximo absoluto esperado.

    Para cada componente c:
        per_tick_c = total_c / tick_count
        norm_c = per_tick_c / max_abs_c

    Resultado:
    - magnitudes comparables entre componentes antes de ponderar.
    - rango aproximado en [-1, 1].
    """
    normalized = {}
    for name, total in totals.items():
        per_tick = total / tick_count
        max_abs = MAX_ABS_COMPONENT_PER_TICK[name]
        normalized[name] = per_tick / max_abs
    return normalized


def component_survival(env, conf):
    """
    Supervivencia tactica basada en amenaza y confianza de acciones.

    Politica por regimen de amenaza:
    - threat > THREAT_HIGH: foco en conductas defensivas.
    - THREAT_LOW < threat <= THREAT_HIGH: transicion tactica.
    - threat <= THREAT_LOW: favorecer exploracion controlada.

    Justificacion:
    - traduce reglas de supervivencia al espacio continuo de confianzas de accion.
    - el clipping final evita que un solo tick distorsione la evaluacion global.
    """
    score = 0.0
    if env.threat > THREAT_HIGH:
        score += MEDIUM_IMPACT * conf.get("dormir", 0.0)
        score += MEDIUM_IMPACT * conf.get("flee", 0.0)
        score -= MEDIUM_IMPACT * conf.get("explore", 0.0)
        score -= SOFT_IMPACT * conf.get("idle", 0.0)
    elif env.threat > THREAT_LOW:
        score += SOFT_IMPACT * conf.get("dormir", 0.0)
        score += SOFT_IMPACT * conf.get("idle", 0.0)
        score -= SOFT_IMPACT * conf.get("explore", 0.0)
    else:
        score += SOFT_IMPACT * conf.get("explore", 0.0)
        score += SOFT_IMPACT * conf.get("idle", 0.0)
        score -= MEDIUM_IMPACT * conf.get("flee", 0.0)

    return clip(score, MAX_ABS_COMPONENT_PER_TICK["survival"])


def component_consistency(env, emotion, conf, prev_conf):
    """
    Coherencia emocional + suavidad temporal de decisiones.

        Subpartes:
        - threat_alignment = 1 - |stress - threat|
        - calm_alignment = 1 - |valence - light|
        - smoothness = 1 - drift, con drift = ||p_t - p_{t-1}||_1 / 2

        Interpretacion:
        - alta puntuacion cuando el estado interno es congruente con entorno y
            las transiciones de decision no son abruptas.
    """
    threat_alignment = 1.0 - abs(emotion.stress - env.threat)
    calm_alignment = 1.0 - abs(emotion.Valence - env.light)

    smoothness = 1.0
    if prev_conf is not None:
        drift = 0.0
        for action in ACTIONS:
            drift += abs(conf.get(action, 0.0) - prev_conf.get(action, 0.0))
        drift /= 2.0
        smoothness = 1.0 - max(0.0, min(1.0, drift))

    score = (
        SOFT_IMPACT * threat_alignment +
        SOFT_IMPACT * calm_alignment +
        SOFT_IMPACT * smoothness
    ) - MEDIUM_IMPACT

    return clip(score, MAX_ABS_COMPONENT_PER_TICK["consistency"])
