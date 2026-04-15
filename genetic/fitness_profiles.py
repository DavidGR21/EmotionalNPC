"""Jueces del AG por personalidad emocional (modelo de Russell)."""

from genetic.fitness_components import clip
from genetic.fitness_constants import MAX_ABS_COMPONENT_PER_TICK, MEDIUM_IMPACT, SOFT_IMPACT, STRONG_IMPACT


def _conf_score(conf, rewards=None, penalties=None):
    """
    Puntaje auxiliar sobre distribucion de acciones.

    Implementa una suma lineal de recompensas y penalizaciones sobre
    la distribucion de confianza por accion:
        delta = sum(reward_a * p(a)) - sum(penalty_a * p(a))
    """
    rewards = rewards or {}
    penalties = penalties or {}
    delta = 0.0

    for action, weight in rewards.items():
        delta += weight * conf.get(action, 0.0)

    for action, weight in penalties.items():
        delta -= weight * conf.get(action, 0.0)

    return clip(delta, MAX_ABS_COMPONENT_PER_TICK["personality"])


def score_calmada(phase, env, emotion, decision, conf):
    """Perfil calmada: prefiere estabilidad, bajo riesgo y descanso controlado."""
    delta = -SOFT_IMPACT * abs(emotion.stress - env.threat)

    if phase == "calm":
        delta += _conf_score(
            conf,
            rewards={"idle": SOFT_IMPACT, "dormir": SOFT_IMPACT},
            penalties={"flee": MEDIUM_IMPACT, "explore": SOFT_IMPACT},
        )
    elif phase == "escalation":
        delta += _conf_score(
            conf,
            rewards={"dormir": MEDIUM_IMPACT},
            penalties={"explore": MEDIUM_IMPACT},
        )
    elif phase == "peak":
        delta += _conf_score(
            conf,
            rewards={"flee": MEDIUM_IMPACT},
            penalties={"idle": STRONG_IMPACT, "explore": STRONG_IMPACT},
        )
    elif phase == "recovery":
        delta += _conf_score(
            conf,
            rewards={"idle": SOFT_IMPACT, "dormir": SOFT_IMPACT},
        )
    elif phase == "curiosity":
        delta += _conf_score(
            conf,
            rewards={"idle": SOFT_IMPACT},
            penalties={"flee": MEDIUM_IMPACT, "explore": SOFT_IMPACT},
        )

    return clip(delta, MAX_ABS_COMPONENT_PER_TICK["personality"])


def score_feliz(phase, env, emotion, decision, conf):
    """Perfil feliz: prioriza exploracion y evita respuestas defensivas."""
    delta = 0.0

    delta -= SOFT_IMPACT * abs(emotion.Valence - 0.85)

    if phase in ["calm", "curiosity", "recovery"]:
        delta += _conf_score(
            conf,
            rewards={"explore": STRONG_IMPACT},
            penalties={"flee": MEDIUM_IMPACT, "dormir": SOFT_IMPACT},
        )
    else:
        delta += _conf_score(
            conf,
            rewards={"explore": SOFT_IMPACT},
            penalties={"flee": SOFT_IMPACT},
        )

    return clip(delta, MAX_ABS_COMPONENT_PER_TICK["personality"])


def score_miedosa(phase, env, emotion, decision, conf):
    """Perfil miedosa: sobrerreacciona al peligro y favorece huida/ocultamiento."""
    delta = 0.0

    if env.threat > 0.15 or env.sound > 0.25 or emotion.stress > 0.55:
        delta += _conf_score(
            conf,
            rewards={"flee": STRONG_IMPACT, "dormir": MEDIUM_IMPACT},
            penalties={"explore": STRONG_IMPACT, "idle": SOFT_IMPACT},
        )
    else:
        delta += _conf_score(
            conf,
            rewards={"dormir": SOFT_IMPACT},
            penalties={"explore": MEDIUM_IMPACT},
        )

    delta += MEDIUM_IMPACT * conf.get("flee", 0.0) * max(0.0, env.threat - 0.6)

    return clip(delta, MAX_ABS_COMPONENT_PER_TICK["personality"])


def score_triste(phase, env, emotion, decision, conf):
    """Perfil triste: baja energia, evita exploracion y tiende a inaccion."""
    delta = 0.0

    delta -= SOFT_IMPACT * abs(emotion.Valence - 0.2)

    if phase in ["calm", "recovery"]:
        delta += _conf_score(
            conf,
            rewards={"idle": MEDIUM_IMPACT, "dormir": SOFT_IMPACT},
            penalties={"explore": STRONG_IMPACT, "flee": SOFT_IMPACT},
        )
    else:
        delta += _conf_score(
            conf,
            rewards={"idle": SOFT_IMPACT},
            penalties={"explore": MEDIUM_IMPACT},
        )

    if env.threat > 0.85 and emotion.stress > 0.75:
        delta += SOFT_IMPACT * conf.get("flee", 0.0)

    return clip(delta, MAX_ABS_COMPONENT_PER_TICK["personality"])


def get_personality_scorer(personality_type="calmada"):
    """
    Retorna la estrategia de scoring de personalidad solicitada.

    Si el tipo no existe, usa score_calmada como fallback seguro.
    """
    scorers = {
        "calmada": score_calmada,
        "feliz": score_feliz,
        "miedosa": score_miedosa,
        "triste": score_triste,
    }
    return scorers.get(personality_type, score_calmada)
