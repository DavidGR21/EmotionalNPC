from models.emotion import Emotion
from models.environment import Environment
from fuzzy.fuzzifier import fuzzify_arousal, fuzzify_valence
from fuzzy.rules import evaluate_rules, normalize_actions
from fuzzy.inference import choose_action

import time
from collections import Counter

params = {
        "w1": 0.7413104201667432,
        "w2": 0.08276221119907848,
        "w3": 0.012461855765321439,
        "w4": 0.21704476675438203,
        "w5": 0.8199247007109596,
        "w6": 0.4737650235532931,
        "w7": 0.3436643765122384,
        "w8": 0.21460749287341574,
        "k1": 0.5870543041059664,
        "k2": 0.05,
        "k3": 0.08411277202514975,
        "dA": 0.18277349218295186,
        "dV": 1.0,
        "ds": 0.7234798494466729
    }

emotion = Emotion()

DECISION_TO_ATTITUDE = {
    "flee": "panico / supervivencia",
    "hide": "defensivo / cauteloso",
    "observe": "alerta / analitico",
    "explore": "curioso / confiado",
    "idle": "neutral / pasivo",
}

REAL_TIME = True
TICK_SECONDS = 1.0
TARGET_ACTIONS = {"idle", "observe", "hide", "explore", "flee"}


def blend(a, b, alpha):
    return a + (b - a) * alpha


def build_environment_timeline():
    timeline = []

    # Fase 1: calma inicial
    for _ in range(5):
        timeline.append(("calm", Environment(sound=0.05, threat=0.05, light=0.95)))

    # Fase 2: subida de tension
    for i in range(7):
        alpha = i / 6
        timeline.append((
            "escalation",
            Environment(
                sound=blend(0.20, 0.90, alpha),
                threat=blend(0.15, 1.00, alpha),
                light=blend(0.80, 0.10, alpha),
            )
        ))

    # Fase 3: pico de peligro
    for _ in range(6):
        timeline.append(("peak", Environment(sound=0.95, threat=1.00, light=0.05)))

    # Fase 4: recuperacion gradual
    for i in range(8):
        alpha = i / 7
        timeline.append((
            "recovery",
            Environment(
                sound=blend(0.80, 0.10, alpha),
                threat=blend(0.90, 0.05, alpha),
                light=blend(0.20, 0.95, alpha),
            )
        ))

    # Fase 5: curiosidad (entorno seguro y bien iluminado, con estimulo moderado)
    for _ in range(8):
        timeline.append(("curiosity", Environment(sound=0.55, threat=0.00, light=1.00)))

    return timeline


def run_realtime_cycle():
    timeline = build_environment_timeline()
    seen_actions = []
    print("=== Simulacion en tiempo real (1 decision por segundo) ===")

    for tick, (phase, env) in enumerate(timeline, start=1):
        emotion.update(env, params)

        af = fuzzify_arousal(emotion.Arousal)
        vf = fuzzify_valence(emotion.Valence)

        actions = normalize_actions(evaluate_rules(af, vf))
        decision = choose_action(actions)
        attitude = DECISION_TO_ATTITUDE.get(decision, "sin clasificacion")
        seen_actions.append(decision)

        sorted_actions = sorted(actions.items(), key=lambda x: x[1], reverse=True)
        top_1 = sorted_actions[0]
        top_2 = sorted_actions[1]

        print("-" * 65)
        print(f"⏱️ TICK: {tick:02d} | 📍 FASE: {phase.upper()}")
        print(f"🌍 ENTORNO: [Sonido: {env.sound:.2f} | Amenaza: {env.threat:.2f} | Luz: {env.light:.2f}]")
        print(f"🧠 EMOCIÓN: [Estrés: {emotion.stress:.3f} | Arousal: {emotion.Arousal:.3f} | Valencia: {emotion.Valence:.3f}]")
        print(f"🤖 ACCIÓN:  >> {decision.upper()} <<  ({attitude})")
        print(f"📊 TOP 2:   {top_1[0].upper()} ({top_1[1]:.2f}) vs {top_2[0].upper()} ({top_2[1]:.2f})")

        if REAL_TIME:
            time.sleep(TICK_SECONDS)

    counts = Counter(seen_actions)
    seen_set = set(seen_actions)
    missing = sorted(TARGET_ACTIONS - seen_set)

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

    print("=== Fin del ciclo ===")


if __name__ == "__main__":
    run_realtime_cycle()

