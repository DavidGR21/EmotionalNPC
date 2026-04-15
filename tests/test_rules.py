from models.emotion import Emotion
from models.environment import Environment
from fuzzy.fuzzifier import fuzzify_arousal, fuzzify_valence
from fuzzy.rules import evaluate_rules, normalize_actions
from fuzzy.inference import choose_action
from controllers.personality_controller import load_personality


PARAMS = {
    "recepcion_sonoro": 0.4,
    "recepcion_amenaza": 0.5,
    "recepcion_nivel_luz": 0.3,
    "recepcion_estres": 0.2,
    "comfort_luz": 0.7,
    "vulnerabilidad_peligro": 0.6,
    "misofonia": 0.5,
    "vulnerabilidad_estres": 0.3,
    "sensibilidad_amenaza": 0.3,
    "sensibilidad_auditiva": 0.2,
    "sensibilidad_oscuridad": 0.2,
    "volatilidad": 0.6,
    "estabilidad_emocional": 0.2,
    "resiliencia_estres": 0.15,
}

EMOTION_STEPS = 10

FUZZY_GENOME = load_personality("explorador")
if not FUZZY_GENOME:
    raise ValueError("No se encontro personalidad entrenada 'normal' en data/personalities.json")


EMOTION_CASES = [
    ("danger", Environment(sound=1.0, threat=1.0, light=0.0)),
    ("calm", Environment(sound=0.0, threat=0.0, light=1.0)),
    ("balanced", Environment(sound=0.5, threat=0.5, light=0.5)),
    ("loud", Environment(sound=1.0, threat=0.0, light=1.0)),
    ("dark", Environment(sound=0.0, threat=1.0, light=0.0)),
    ("silent_threat", Environment(sound=0.0, threat=1.0, light=1.0)),
    ("noisy_safe", Environment(sound=1.0, threat=0.0, light=0.0)),
    ("dim_tension", Environment(sound=0.4, threat=0.7, light=0.2)),
    ("bright_alarm", Environment(sound=0.8, threat=0.9, light=0.9)),
    ("stealth_zone", Environment(sound=0.1, threat=0.3, light=0.1)),
    ("crowded_corridor", Environment(sound=0.9, threat=0.6, light=0.6)),
    ("foggy_night", Environment(sound=0.2, threat=0.8, light=0.1)),
    ("safe_but_dark", Environment(sound=0.0, threat=0.0, light=0.0)),
    ("safe_but_noisy", Environment(sound=0.8, threat=0.0, light=1.0)),
    ("critical_threat", Environment(sound=0.6, threat=1.0, light=0.1)),
]


FUZZY_CASES = [
    ("flee", 0.9, 0.1),
    ("hide", 0.5, 0.1),
    ("idle_neg", 0.1, 0.1),
    ("observe1", 0.9, 0.5),
    ("observe2", 0.5, 0.5),
    ("observe3", 0.1, 0.5),
    ("explore1", 0.9, 0.9),
    ("explore2", 0.5, 0.9),
    ("idle_pos", 0.1, 0.9),
]


def compute_emotion_step(environment, params):
    stimulus_s = (
        params["sensibilidad_amenaza"] * environment.threat +
        params["sensibilidad_auditiva"] * environment.sound +
        params["sensibilidad_oscuridad"] * (1 - environment.light)
    ) / (
        params["sensibilidad_amenaza"] +
        params["sensibilidad_auditiva"] +
        params["sensibilidad_oscuridad"]
    )

    stress = stimulus_s * params["resiliencia_estres"]

    target_arousal = (
        params["recepcion_sonoro"] * environment.sound +
        params["recepcion_amenaza"] * environment.threat +
        params["recepcion_nivel_luz"] * (1 - environment.light) +
        params["recepcion_estres"] * stress
    ) / (
        params["recepcion_sonoro"] +
        params["recepcion_amenaza"] +
        params["recepcion_nivel_luz"] +
        params["recepcion_estres"]
    )

    target_valence = (
        params["comfort_luz"] * environment.light -
        (
            params["vulnerabilidad_peligro"] * environment.threat +
            params["misofonia"] * environment.sound +
            params["vulnerabilidad_estres"] * stress
        )
    )
    target_valence = max(-1, min(1, target_valence))
    target_valence = (target_valence + 1) / 2

    expected_arousal = target_arousal * params["volatilidad"]
    expected_valence = target_valence * params["estabilidad_emocional"]

    emotion = Emotion()
    emotion.update(environment, params)

    return {
        "stimulus_s": stimulus_s,
        "stress": stress,
        "target_arousal": target_arousal,
        "target_valence": target_valence,
        "expected_arousal": expected_arousal,
        "expected_valence": expected_valence,
        "emotion": emotion,
    }


def simulate_emotion_steps(environment, params, steps):
    emotion = Emotion()
    history = []

    for step in range(1, steps + 1):
        emotion.update(environment, params)
        af = fuzzify_arousal(emotion.Arousal, FUZZY_GENOME)
        vf = fuzzify_valence(emotion.Valence, FUZZY_GENOME)
        actions = normalize_actions(evaluate_rules(af, vf))
        decision = choose_action(actions)

        history.append({
            "step": step,
            "stress": emotion.stress,
            "arousal": emotion.Arousal,
            "valence": emotion.Valence,
            "decision": decision,
        })

    return history


def run_emotion_case(case_name, environment, params, steps=EMOTION_STEPS):
    result = compute_emotion_step(environment, params)
    emotion = result["emotion"]

    tolerance = 1e-9
    assert abs(emotion.Arousal - result["expected_arousal"]) < tolerance, case_name
    assert abs(emotion.Valence - result["expected_valence"]) < tolerance, case_name

    history = simulate_emotion_steps(environment, params, steps)

    print(f"Emotion case: {case_name}")
    print(
        "  Environment -> "
        f"sound={environment.sound:.2f}, threat={environment.threat:.2f}, light={environment.light:.2f}"
    )
    print(f"  Stress mix   -> {result['stimulus_s']:.4f}")
    print(f"  Stress       -> {result['stress']:.4f}")
    print(f"  Target A     -> {result['target_arousal']:.4f}")
    print(f"  Target V     -> {result['target_valence']:.4f}")
    print(f"  Emotion A(1) -> {emotion.Arousal:.4f}")
    print(f"  Emotion V(1) -> {emotion.Valence:.4f}")
    print("  Evolution    ->")
    for row in history:
        print(
            f"    step {row['step']:02d}: "
            f"stress={row['stress']:.4f}, "
            f"A={row['arousal']:.4f}, "
            f"V={row['valence']:.4f}, "
            f"decision={row['decision']}"
        )
    print()


def run_fuzzy_case(expected_rule, arousal, valence):
    af = fuzzify_arousal(arousal, FUZZY_GENOME)
    vf = fuzzify_valence(valence, FUZZY_GENOME)
    actions = normalize_actions(evaluate_rules(af, vf))
    decision = choose_action(actions)

    print(f"Fuzzy case: {expected_rule}")
    print(f"  Input  -> Arousal={arousal:.2f}, Valence={valence:.2f}")
    print(f"  Af     -> {af}")
    print(f"  Vf     -> {vf}")
    print(f"  Actions-> {actions}")
    print(f"  Decide -> {decision}")
    print()


if __name__ == "__main__":
    print("=== Emotion calculations ===")
    for case_name, environment in EMOTION_CASES:
        run_emotion_case(case_name, environment, PARAMS, EMOTION_STEPS)

    # print("=== Fuzzy rule cases ===")
    # for expected_rule, arousal, valence in FUZZY_CASES:
    #     run_fuzzy_case(expected_rule, arousal, valence)