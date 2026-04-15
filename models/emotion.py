from utils.math_utils import clamp
class Emotion:
    def __init__(self, environment=None, params=None):
        self.stress = 0.0
        self.Arousal = 0.0
        self.Valence = 0.0

    def update(self, environment, params):
        # 1. STRESS (dinámico)
        stimulus_s = (
            params["sensibilidad_amenaza"] * environment.threat +
            params["sensibilidad_auditiva"] * environment.sound +
            params["sensibilidad_oscuridad"] * (1 - environment.light)
        ) / (
            params["sensibilidad_amenaza"] +
            params["sensibilidad_auditiva"] +
            params["sensibilidad_oscuridad"]
        )

        self.stress += (stimulus_s - self.stress) * params["resiliencia_estres"]

        # 2. AROUSAL (reactivo)
        target_A = (
            params["recepcion_sonoro"] * environment.sound +
            params["recepcion_amenaza"] * environment.threat +
            params["recepcion_nivel_luz"] * (1 - environment.light) +
            params["recepcion_estres"] * self.stress
        ) / (
            params["recepcion_sonoro"] +
            params["recepcion_amenaza"] +
            params["recepcion_nivel_luz"] +
            params["recepcion_estres"]
        )

        self.Arousal += (target_A - self.Arousal) * params["volatilidad"]

        # 3. VALENCE (SIN NORMALIZAR MAL)
        target_V = (
            params["comfort_luz"] * environment.light -
            (
                params["vulnerabilidad_peligro"] * environment.threat +
                params["misofonia"] * environment.sound +
                params["vulnerabilidad_estres"] * self.stress
            )
        )

        target_V = max(-1, min(1, target_V))   # rango [-1, 1]

        # Convertir a [0,1] 
        target_V = (target_V + 1) / 2

        self.Valence += (target_V - self.Valence) * params["estabilidad_emocional"]

        # 4. CLAMP FINAL
        self.stress = clamp(self.stress)
        self.Arousal = clamp(self.Arousal)
        self.Valence = clamp(self.Valence)

        # print(f"Estres: {self.stress} Arousal: {self.Arousal} Valence: {self.Valence}")