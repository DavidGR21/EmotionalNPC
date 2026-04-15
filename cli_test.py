from models.emotion import Emotion
from models.environment import Environment
from fuzzy.engine import process_fuzzy_logic
from models.enums import get_attitude_from_decision
from controllers.personality_controller import load_personality

import time
from collections import Counter

# Semilla biológica extraída del entrenamiento genético
params = load_personality("explorador")
if not params:
    raise ValueError("No se encontro personalidad entrenada 'explorador' en data/personalities.json")

emotion = Emotion()

REAL_TIME = True
TICK_SECONDS = 1.0
TARGET_ACTIONS = {"idle", "observe", "hide", "explore", "flee"}
TARGET_QUADRANTS = {"Q1_high_pos", "Q2_high_neg", "Q3_low_neg", "Q4_low_pos"}

def blend(a, b, alpha):
    return a + (b - a) * alpha


def add_hold(timeline, phase, ticks, sound, threat, light):
    for _ in range(ticks):
        timeline.append((phase, Environment(sound=sound, threat=threat, light=light)))


def add_ramp(timeline, phase, ticks, start_env, end_env):
    for i in range(ticks):
        alpha = i / max(1, ticks - 1)
        timeline.append((
            phase,
            Environment(
                sound=blend(start_env[0], end_env[0], alpha),
                threat=blend(start_env[1], end_env[1], alpha),
                light=blend(start_env[2], end_env[2], alpha),
            ),
        ))


def classify_russell_quadrant(arousal, valence, split=0.5):
    if arousal >= split and valence >= split:
        return "Q1_high_pos"
    if arousal >= split and valence < split:
        return "Q2_high_neg"
    if arousal < split and valence < split:
        return "Q3_low_neg"
    return "Q4_low_pos"


def evaluate_quadrant_coverage_for_params(test_params):
    emotion = Emotion()
    quadrants = []

    for _, env in build_environment_timeline():
        emotion.update(env, test_params)
        quadrants.append(classify_russell_quadrant(emotion.Arousal, emotion.Valence))

    return Counter(quadrants)


def run_quadrant_probe(personality_names):
    print("=== Sonda Russell (multi-personalidad) ===")
    global_seen = set()

    for personality_name in personality_names:
        test_params = load_personality(personality_name)
        if not test_params:
            print(f"  {personality_name}: no disponible")
            continue

        counts = evaluate_quadrant_coverage_for_params(test_params)
        seen = set(counts.keys())
        global_seen |= seen
        missing = sorted(TARGET_QUADRANTS - seen)

        print(f"  {personality_name}: {dict(counts)}")
        if missing:
            print(f"    faltan: {', '.join(missing)}")
        else:
            print("    cobertura completa")

    global_missing = sorted(TARGET_QUADRANTS - global_seen)
    if global_missing:
        print(f"Cobertura global incompleta: {', '.join(global_missing)}")
    else:
        print("Cobertura global completa de los 4 cuadrantes.")

def build_environment_timeline():
    timeline = []

    # Q4 (bajo arousal, valencia positiva): calma/seguridad.
    add_hold(timeline, "calm", 5, sound=0.03, threat=0.02, light=0.98)

    # Q2 (alto arousal, valencia negativa): escalada/pico hostil.
    add_ramp(
        timeline,
        "escalation",
        6,
        start_env=(0.55, 0.20, 0.80),
        end_env=(0.98, 1.00, 0.03),
    )
    add_hold(timeline, "peak", 5, sound=0.98, threat=1.00, light=0.02)

    # Q1 (alto arousal, valencia positiva): rebote post-crisis.
    # Se aprovecha el arousal residual del pico y se sube valencia con luz alta
    # y amenaza/sonido bajos.
    add_ramp(
        timeline,
        "curiosity",
        5,
        start_env=(0.65, 0.60, 0.20),
        end_env=(0.12, 0.02, 1.00),
    )
    add_hold(timeline, "curiosity", 5, sound=0.10, threat=0.02, light=1.00)

    # Q3 (bajo arousal, valencia negativa): fatiga/apagado tras crisis.
    add_ramp(
        timeline,
        "recovery",
        6,
        start_env=(0.85, 0.85, 0.12),
        end_env=(0.07, 0.35, 0.03),
    )
    add_hold(timeline, "recovery", 4, sound=0.04, threat=0.30, light=0.03)

    # Retorno a Q4 para comprobar adaptacion y estabilidad final.
    add_ramp(
        timeline,
        "recovery",
        5,
        start_env=(0.20, 0.25, 0.20),
        end_env=(0.03, 0.03, 0.98),
    )
    add_hold(timeline, "calm", 4, sound=0.03, threat=0.03, light=0.98)

    return timeline

def run_realtime_cycle():
    timeline = build_environment_timeline()
    seen_actions = []
    seen_quadrants = []
    print("=== Simulacion en tiempo real (1 decision por segundo) ===")

    for tick, (phase, env) in enumerate(timeline, start=1):
        # 1. Update Continuous System
        emotion.update(env, params)

        # 2. Pipeline Mental Desacoplado
        decision, top_opts = process_fuzzy_logic(emotion, params)
        attitude = get_attitude_from_decision(decision)
        quadrant = classify_russell_quadrant(emotion.Arousal, emotion.Valence)
        seen_actions.append(decision)
        seen_quadrants.append(quadrant)

        # Imprimir bonita métrica
        top_list = list(top_opts.items())
        top_1 = top_list[0] if len(top_list) > 0 else ("none", 0)
        top_2 = top_list[1] if len(top_list) > 1 else ("none", 0)

        print("-" * 65)
        print(f"⏱️ TICK: {tick:02d} | 📍 FASE: {phase.upper()}")
        print(f"🌍 ENTORNO: [Sonido: {env.sound:.2f} | Amenaza: {env.threat:.2f} | Luz: {env.light:.2f}]")
        print(f"🧠 EMOCIÓN: [Estrés: {emotion.stress:.3f} | Arousal: {emotion.Arousal:.3f} | Valencia: {emotion.Valence:.3f}]")
        print(f"🧭 RUSSELL: {quadrant}")
        print(f"🤖 ACCIÓN:  >> {decision.upper()} <<  ({attitude})")
        print(f"📊 TOP 2:   {top_1[0].upper()} ({top_1[1]:.2f}) vs {top_2[0].upper()} ({top_2[1]:.2f})")

        if REAL_TIME:
            time.sleep(TICK_SECONDS)

    counts = Counter(seen_actions)
    seen_set = set(seen_actions)
    missing = sorted(TARGET_ACTIONS - seen_set)

    quadrant_counts = Counter(seen_quadrants)
    seen_quadrants_set = set(seen_quadrants)
    missing_quadrants = sorted(TARGET_QUADRANTS - seen_quadrants_set)

    print("=== Resumen de acciones ===")
    for action in sorted(TARGET_ACTIONS):
        print(f"  {action}: {counts.get(action, 0)}")

    if missing:
        print(f"Acciones no activadas: {', '.join(missing)}")
        print(
            "Nota: 'explore' requiere valence positiva y arousal medio/alto al mismo tiempo; "
            "con los pesos actuales, ese cruce es raro en una dinamica natural."
        )
    else:
        print("Se activaron todas las acciones objetivo.")

    print("=== Cobertura Russell ===")
    for quadrant in sorted(TARGET_QUADRANTS):
        print(f"  {quadrant}: {quadrant_counts.get(quadrant, 0)}")

    if missing_quadrants:
        print(f"Cuadrantes no cubiertos: {', '.join(missing_quadrants)}")
    else:
        print("Se cubrieron los 4 cuadrantes del circumplejo de Russell.")

    run_quadrant_probe(["normal", "cobarde", "valiente", "explorador", "superviviente"])

    print("=== Fin del ciclo ===")


if __name__ == "__main__":
    run_realtime_cycle()

