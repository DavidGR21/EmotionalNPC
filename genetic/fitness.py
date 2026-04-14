from models.emotion import Emotion
from fuzzy.engine import process_fuzzy_logic
from cli_test import build_environment_timeline


SOFT_IMPACT = 5.0
MEDIUM_IMPACT = 10.0
STRONG_IMPACT = 20.0

MAX_ABS_COMPONENT_PER_TICK = {
    "survival": STRONG_IMPACT,
    "consistency": MEDIUM_IMPACT,
    "personality": STRONG_IMPACT,
}

COMPONENT_WEIGHTS = {
    "survival": 0.35,
    "consistency": 0.25,
    "personality": 0.40,
}


def _fuzzy_regularization(genome):
    """
    Penaliza configuraciones difusas degeneradas para estabilizar el aprendizaje.
    """
    penalty = 0.0

    cA_low = genome.get("cA_low", 0.1)
    cA_medium = genome.get("cA_medium", 0.5)
    cA_high = genome.get("cA_high", 0.9)
    cV_negative = genome.get("cV_negative", 0.1)
    cV_neutral = genome.get("cV_neutral", 0.5)
    cV_positive = genome.get("cV_positive", 0.9)

    a_gap_1 = cA_medium - cA_low
    a_gap_2 = cA_high - cA_medium
    v_gap_1 = cV_neutral - cV_negative
    v_gap_2 = cV_positive - cV_neutral

    min_gap = 0.08
    for gap in [a_gap_1, a_gap_2, v_gap_1, v_gap_2]:
        if gap < min_gap:
            penalty += (min_gap - gap) * 40.0

    sigma = genome.get("sigma", 0.15)
    if sigma < 0.07:
        penalty += (0.07 - sigma) * 60.0
    if sigma > 0.30:
        penalty += (sigma - 0.30) * 40.0

    return penalty


def _annealed_regularization(genome, generation, total_generations):
    """
    Regularización adaptativa: fuerte al inicio, suave al final.
    """
    raw_penalty = _fuzzy_regularization(genome)
    total = max(1, total_generations)
    progress = (max(1, generation) - 1) / max(1, total - 1)
    weight = 1.0 - 0.75 * progress
    return raw_penalty * weight


def _clip(value, limit):
    return max(-limit, min(limit, value))


def _component_survival(env, conf):
    """
    Supervivencia táctica basada en amenaza y confianza de acciones.
    """
    score = 0.0
    if env.threat > 0.7:
        score += MEDIUM_IMPACT * conf.get("hide", 0.0)
        score += MEDIUM_IMPACT * conf.get("flee", 0.0)
        score -= MEDIUM_IMPACT * conf.get("explore", 0.0)
        score -= SOFT_IMPACT * conf.get("idle", 0.0)
    elif env.threat > 0.2:
        score += SOFT_IMPACT * conf.get("observe", 0.0)
        score += SOFT_IMPACT * conf.get("hide", 0.0)
        score -= SOFT_IMPACT * conf.get("explore", 0.0)
    else:
        score += SOFT_IMPACT * conf.get("explore", 0.0)
        score += SOFT_IMPACT * conf.get("observe", 0.0)
        score -= MEDIUM_IMPACT * conf.get("flee", 0.0)

    return _clip(score, MAX_ABS_COMPONENT_PER_TICK["survival"])


def _component_consistency(env, emotion, conf, prev_conf):
    """
    Coherencia emocional + suavidad temporal de decisiones.
    """
    threat_alignment = 1.0 - abs(emotion.stress - env.threat)
    calm_alignment = 1.0 - abs(emotion.Valence - env.light)

    smoothness = 1.0
    if prev_conf is not None:
        drift = 0.0
        for action in ["flee", "hide", "observe", "explore", "idle"]:
            drift += abs(conf.get(action, 0.0) - prev_conf.get(action, 0.0))
        drift /= 2.0
        smoothness = 1.0 - max(0.0, min(1.0, drift))

    score = (
        SOFT_IMPACT * threat_alignment +
        SOFT_IMPACT * calm_alignment +
        SOFT_IMPACT * smoothness
    ) - MEDIUM_IMPACT

    return _clip(score, MAX_ABS_COMPONENT_PER_TICK["consistency"])

# ==========================================
# EVALUADOR DEL DESEMPEÑO (FITNESS FUNCTION)
# ==========================================
# Aquí es donde se juzga si un NPC sobrevivirá a la selección natural.
# Usamos el Patrón de Diseño "Factory" (Fábrica): Dependiendo de qué "palabra" le
# pasemos a la función (ej: "cobarde"), esta nos construirá y devolverá una min-función
# de evaluación que premia cosas distintas.

def get_fitness_evaluator(personality_type="normal"):
    """
    Factory que retorna la función de evaluación (fitness) correspondiente
    a la personalidad que queremos generar.
    """
    
    def evaluate_timeline(genome, scoring_logic, generation=1, total_generations=1):
        """
        Wrapper base o "Simulador Universal".
        En lugar de repetir el bucle de la línea del tiempo en cada personalidad,
        esta función corre el simulador base usando los genes recibidos (genome) y 
        simplemente le pregunta a la función `scoring_logic` de abajo cuántos puntos
        se merece en cada instante (tick) simulado.
        """
        # 1. Instanciamos un cerebro emocional limpio exclusivo para este NPC
        emotion = Emotion()
        # 2. Obtenemos el entorno hostil de prueba (el laberinto)
        timeline = build_environment_timeline()
        totals = {
            "survival": 0.0,
            "consistency": 0.0,
            "personality": 0.0,
        }
        prev_conf = None
        
        # 3. Soltamos al NPC a la arena y simulamos el paso del tiempo
        for tick, (phase, env) in enumerate(timeline):
            # El NPC actualiza sus emociones basado en sus "genes"
            emotion.update(env, genome)
            
            # 4. Invocamos al motor de Lógica Difusa centralizado
            decision, _, conf = process_fuzzy_logic(emotion, genome, include_action_scores=True)
            
            # 5. PASO CRITICO: Le preguntamos al juez qué opina de esa decisión.
            totals["survival"] += _component_survival(env, conf)
            totals["consistency"] += _component_consistency(env, emotion, conf, prev_conf)
            totals["personality"] += scoring_logic(phase, env, emotion, decision, conf)
            prev_conf = conf

        tick_count = max(1, len(timeline))
        normalized = {}
        for name, total in totals.items():
            per_tick = total / tick_count
            max_abs = MAX_ABS_COMPONENT_PER_TICK[name]
            normalized[name] = per_tick / max_abs

        score = 0.0
        for name, weight in COMPONENT_WEIGHTS.items():
            score += weight * normalized[name]

        reg_penalty = _annealed_regularization(genome, generation, total_generations)
        score -= reg_penalty / 100.0
            
        return score

    def _conf_score(conf, rewards=None, penalties=None):
        rewards = rewards or {}
        penalties = penalties or {}
        delta = 0.0

        for action, weight in rewards.items():
            delta += weight * conf.get(action, 0.0)

        for action, weight in penalties.items():
            delta -= weight * conf.get(action, 0.0)

        return _clip(delta, MAX_ABS_COMPONENT_PER_TICK["personality"])

    # === DEFINICIÓN DE PERFILES PSICOLÓGICOS (SCORING LOGICS / JUECES) ===
    # Cada una de estas funciones es un juez con diferentes gustos. Retornan los "Puntos"
    # que ganará o perderá el individuo en el segundo actual de la simulación temporal.

    def _score_normal(phase, env, emotion, decision, conf):
        """El Juez Normal: Busca coherencia realista paso a paso."""
        delta = -SOFT_IMPACT * abs(emotion.stress - env.threat)
        
        if phase == "calm":                 # Si no hay peligro...
            delta += _conf_score(
                conf,
                rewards={"idle": SOFT_IMPACT, "observe": SOFT_IMPACT},
                penalties={"flee": MEDIUM_IMPACT, "hide": SOFT_IMPACT},
            )
        elif phase == "escalation":         # Si las cosas se calientan...
            delta += _conf_score(
                conf,
                rewards={"observe": MEDIUM_IMPACT, "hide": SOFT_IMPACT},
                penalties={"explore": SOFT_IMPACT},
            )
        elif phase == "peak":               # Ante peligro de muerte inminente...
            delta += _conf_score(
                conf,
                rewards={"flee": MEDIUM_IMPACT, "hide": MEDIUM_IMPACT},
                penalties={"idle": STRONG_IMPACT, "explore": STRONG_IMPACT},
            )
        elif phase == "recovery":           # Tras escapar...
            delta += _conf_score(
                conf,
                rewards={"observe": SOFT_IMPACT, "hide": SOFT_IMPACT},
            )
        elif phase == "curiosity":          # Si el lugar está bien iluminado y seguro...
            delta += _conf_score(
                conf,
                rewards={"explore": MEDIUM_IMPACT},
                penalties={"flee": MEDIUM_IMPACT},
            )
        return _clip(delta, MAX_ABS_COMPONENT_PER_TICK["personality"])

    
    def _score_valiente(phase, env, emotion, decision, conf):
        """El Juez Temerario: Te castiga por huir a menos que literalmente vayas a morir."""
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
            
        return _clip(delta, MAX_ABS_COMPONENT_PER_TICK["personality"])

    def _score_explorador(phase, env, emotion, decision, conf):
        """El Juez Curioso: Te premia si saltas a la aventura incluso estando asustado, excepto si ves la amenaza en tu cara."""
        delta = 0.0
        
        # Condición Especial (La curiosidad mató al gato): 
        # Alto estrés (tal vez por el silencio aterrador o la oscuridad), PERO no hay amenaza real evidente (<= 0.2).
        if emotion.stress > 0.6 and env.threat <= 0.2:
            delta += _conf_score(
                conf,
                rewards={"explore": STRONG_IMPACT},
                penalties={"hide": MEDIUM_IMPACT},
            )
        else:
            # Si sí hay una amenaza letal latente evidente frente a él -> Comportamiento normal esconderse
            if env.threat > 0.5:
                delta += _conf_score(
                    conf,
                    rewards={"hide": SOFT_IMPACT},
                    penalties={"explore": STRONG_IMPACT},
                )
            else:
                # Regla general del explorador para el resto de fases
                delta += _conf_score(
                    conf,
                    rewards={"explore": MEDIUM_IMPACT},
                    penalties={"idle": SOFT_IMPACT},
                )
        return _clip(delta, MAX_ABS_COMPONENT_PER_TICK["personality"])

    def _score_cobarde(phase, env, emotion, decision, conf):
        """El Juez Paranoico: Te premia si huyes de tu propia sombra."""
        delta = 0.0
        # Reacciona exageradamente a todo lo que suba el stress
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
            
        return _clip(delta, MAX_ABS_COMPONENT_PER_TICK["personality"])

    def _score_superviviente(phase, env, emotion, decision, conf):
        """El Juez Superviviente: Táctico e implacable. Prioriza salvaguardar la vida, pero no es paralizado por el miedo."""
        delta = 0.0
        
        # Supervivencia al límite: Penalización si el estrés llega a niveles fatales y no hace nada consecuente.
        if emotion.stress > 0.9:
            delta += _conf_score(
                conf,
                rewards={"hide": MEDIUM_IMPACT, "flee": MEDIUM_IMPACT},
                penalties={"idle": STRONG_IMPACT, "explore": STRONG_IMPACT},
            )
            
        # Reacción objetiva a las amenazas sin depender de qué fase estemos arbitrariamente:
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
            
        return _clip(delta, MAX_ABS_COMPONENT_PER_TICK["personality"])
        
    # ==========================================
    # EL MOTOR DE LA FÁBRICA (INYECCION DEL JUEZ)
    # ==========================================
    # Retornamos una función lambda "Lista para usar" (Closure). 
    # Cuando train_ga.py llame a esta lambda con sus genes `genome`,
    # ejecutará nuestro simulador pasándole el juez específico que fue pedido.
    if personality_type == "cobarde":
        return lambda genome, generation=1, total_generations=1: evaluate_timeline(
            genome, _score_cobarde, generation, total_generations
        )
    elif personality_type == "valiente":
        return lambda genome, generation=1, total_generations=1: evaluate_timeline(
            genome, _score_valiente, generation, total_generations
        )
    elif personality_type == "explorador":
        return lambda genome, generation=1, total_generations=1: evaluate_timeline(
            genome, _score_explorador, generation, total_generations
        )
    elif personality_type == "superviviente":
        return lambda genome, generation=1, total_generations=1: evaluate_timeline(
            genome, _score_superviviente, generation, total_generations
        )
    else:
        return lambda genome, generation=1, total_generations=1: evaluate_timeline(
            genome, _score_normal, generation, total_generations
        )