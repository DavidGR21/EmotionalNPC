"""
Constantes y configuracion del fitness de EmotionalNPC.

Este modulo centraliza los hiperparametros del objetivo de optimizacion para:
1) Evitar numeros magicos dispersos.
2) Facilitar trazabilidad experimental.
3) Mantener coherencia entre componentes del fitness.

Justificacion general:
- Se usan impactos discretos (5, 10, 20) para expresar importancia relativa.
- Se normalizan componentes por maximos absolutos por tick para que compartan
    escala aproximada en [-1, 1] antes de ponderarse.
"""

SOFT_IMPACT = 5.0
MEDIUM_IMPACT = 10.0
STRONG_IMPACT = 20.0

# Umbrales de amenaza para cambiar de politica tactica.
# 0.2 separa zona segura de tension moderada.
# 0.7 separa tension moderada de peligro alto.

THREAT_LOW = 0.2
THREAT_HIGH = 0.7

MIN_CENTER_GAP = 0.08
GAP_PENALTY_FACTOR = 40.0
SIGMA_LOW = 0.07
SIGMA_HIGH = 0.30
SIGMA_LOW_PENALTY = 60.0
SIGMA_HIGH_PENALTY = 40.0

# Annealing de regularizacion: al inicio prioriza estructura; al final,
# ajuste fino de desempeno.

ANNEAL_START = 1.0
ANNEAL_END = 0.25

ACTIONS = ("flee", "dormir", "explore", "idle")

MAX_ABS_COMPONENT_PER_TICK = {
    "survival": STRONG_IMPACT,
    "consistency": MEDIUM_IMPACT,
    "personality": STRONG_IMPACT,
}

#  escalarizacion multiobjetivo.
# Se prioriza personalidad (0.40), luego supervivencia (0.35),
# y finalmente consistencia (0.25).
COMPONENT_WEIGHTS = {
    "survival": 0.35,
    "consistency": 0.25,
    "personality": 0.40,
}
