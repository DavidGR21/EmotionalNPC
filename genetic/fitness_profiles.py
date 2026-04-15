"""
Estrategias de perfil psicologico para el componente de personalidad.

Diseno:
- Todas las estrategias viven en un unico modulo para facilitar comparacion,
  mantenimiento y trazabilidad experimental.
- Cada estrategia retorna un delta por tick y se recorta al rango del componente.
"""

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


def score_normal(phase, env, emotion, decision, conf):
    """
    Perfil baseline: coherencia situacional y emocional por fases narrativas.
    """
    delta = -SOFT_IMPACT * abs(emotion.stress - env.threat)

    if phase == "calm":
        delta += _conf_score(
            conf,
            rewards={"idle": SOFT_IMPACT, "observe": SOFT_IMPACT},
            penalties={"flee": MEDIUM_IMPACT, "hide": SOFT_IMPACT},
        )
    elif phase == "escalation":
        delta += _conf_score(
            conf,
            rewards={"observe": MEDIUM_IMPACT, "hide": SOFT_IMPACT},
            penalties={"explore": SOFT_IMPACT},
        )
    elif phase == "peak":
        delta += _conf_score(
            conf,
            rewards={"flee": MEDIUM_IMPACT, "hide": MEDIUM_IMPACT},
            penalties={"idle": STRONG_IMPACT, "explore": STRONG_IMPACT},
        )
    elif phase == "recovery":
        delta += _conf_score(
            conf,
            rewards={"observe": SOFT_IMPACT, "hide": SOFT_IMPACT},
        )
    elif phase == "curiosity":
        delta += _conf_score(
            conf,
            rewards={"explore": MEDIUM_IMPACT},
            penalties={"flee": MEDIUM_IMPACT},
        )

    return clip(delta, MAX_ABS_COMPONENT_PER_TICK["personality"])


def score_valiente(phase, env, emotion, decision, conf):
    """
    Perfil temerario: penaliza huir salvo condiciones extremas de estres.
    """
    delta = 0.0

    if emotion.stress > 0.95:
        delta += SOFT_IMPACT * conf.get("flee", 0.0)
    else:
        delta -= STRONG_IMPACT * conf.get("flee", 0.0)

    if phase in ["peak", "escalation"]:
        delta += _conf_score(
            conf,
            rewards={"observe": MEDIUM_IMPACT, "idle": SOFT_IMPACT},
            penalties={"hide": SOFT_IMPACT},
        )
    else:
        delta += _conf_score(conf, rewards={"explore": MEDIUM_IMPACT})

    return clip(delta, MAX_ABS_COMPONENT_PER_TICK["personality"])


def score_explorador(phase, env, emotion, decision, conf):
    """
    Perfil curioso: prioriza exploracion, salvo amenaza alta claramente observable.
    """
    delta = 0.0

    if emotion.stress > 0.6 and env.threat <= 0.2:
        delta += _conf_score(
            conf,
            rewards={"explore": STRONG_IMPACT},
            penalties={"hide": MEDIUM_IMPACT},
        )
    else:
        if env.threat > 0.5:
            delta += _conf_score(
                conf,
                rewards={"hide": SOFT_IMPACT},
                penalties={"explore": STRONG_IMPACT},
            )
        else:
            delta += _conf_score(
                conf,
                rewards={"explore": MEDIUM_IMPACT},
                penalties={"idle": SOFT_IMPACT},
            )

    return clip(delta, MAX_ABS_COMPONENT_PER_TICK["personality"])


def score_cobarde(phase, env, emotion, decision, conf):
    """
    Perfil defensivo: sobrerreacciona a amenaza/ruido y favorece ocultarse/huir.
    """
    delta = 0.0

    if env.threat > 0.1 or env.sound > 0.3:
        delta += _conf_score(
            conf,
            rewards={"hide": MEDIUM_IMPACT, "flee": MEDIUM_IMPACT},
            penalties={"explore": STRONG_IMPACT},
        )
    else:
        delta += _conf_score(
            conf,
            rewards={"hide": SOFT_IMPACT},
            penalties={"explore": SOFT_IMPACT},
        )

    if emotion.stress >= 0.8:
        delta += MEDIUM_IMPACT * conf.get("flee", 0.0)

    return clip(delta, MAX_ABS_COMPONENT_PER_TICK["personality"])


def score_superviviente(phase, env, emotion, decision, conf):
    """
    Perfil tactico: prioriza preservar vida con reaccion adaptativa por amenaza.
    """
    delta = 0.0

    if emotion.stress > 0.9:
        delta += _conf_score(
            conf,
            rewards={"hide": MEDIUM_IMPACT, "flee": MEDIUM_IMPACT},
            penalties={"idle": STRONG_IMPACT, "explore": STRONG_IMPACT},
        )

    if env.threat > 0.7:
        delta += _conf_score(
            conf,
            rewards={"flee": MEDIUM_IMPACT, "hide": MEDIUM_IMPACT},
            penalties={"idle": STRONG_IMPACT, "explore": STRONG_IMPACT},
        )
    elif env.threat > 0.2:
        delta += _conf_score(
            conf,
            rewards={"observe": MEDIUM_IMPACT},
            penalties={"explore": MEDIUM_IMPACT},
        )
    else:
        delta += _conf_score(
            conf,
            rewards={"explore": MEDIUM_IMPACT},
            penalties={"idle": SOFT_IMPACT},
        )

    return clip(delta, MAX_ABS_COMPONENT_PER_TICK["personality"])


def get_personality_scorer(personality_type="normal"):
    """
    Retorna la estrategia de scoring de personalidad solicitada.

    Si el tipo no existe, usa score_normal como fallback seguro.
    """
    scorers = {
        "normal": score_normal,
        "cobarde": score_cobarde,
        "valiente": score_valiente,
        "explorador": score_explorador,
        "superviviente": score_superviviente,
    }
    return scorers.get(personality_type, score_normal)
