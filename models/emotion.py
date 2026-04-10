from utils.math_utils import clamp
class Emotion:
    def __init__(self, environment=None, params=None):
        self.stress = 0.0
        self.Arousal = 0.0
        self.Valence = 0.0

    def update(self, environment, params):
        # 1. STRESS (dinámico)
        stimulus_s = (
            params["k1"] * environment.threat +
            params["k2"] * environment.sound +
            params["k3"] * (1 - environment.light)
        ) / (params["k1"] + params["k2"] + params["k3"])

        self.stress += (stimulus_s - self.stress) * params["ds"]

        # 2. AROUSAL (reactivo)
        target_A = (
            params["w1"] * environment.sound +
            params["w2"] * environment.threat +
            params["w3"] * (1 - environment.light) +
            params["w4"] * self.stress
        ) / (params["w1"] + params["w2"] + params["w3"] + params["w4"])

        self.Arousal += (target_A - self.Arousal) * params["dA"]

        # 3. VALENCE (SIN NORMALIZAR MAL)
        target_V = (
            params["w5"] * environment.light -
            (
                params["w6"] * environment.threat +
                params["w7"] * environment.sound +
                params["w8"] * self.stress
            )
        )

        target_V = max(-1, min(1, target_V))   # rango [-1, 1]

        # Convertir a [0,1] 
        target_V = (target_V + 1) / 2

        self.Valence += (target_V - self.Valence) * params["dV"]

        # 4. CLAMP FINAL
        self.stress = clamp(self.stress)
        self.Arousal = clamp(self.Arousal)
        self.Valence = clamp(self.Valence)

        # print(f"Estres: {self.stress} Arousal: {self.Arousal} Valence: {self.Valence}")