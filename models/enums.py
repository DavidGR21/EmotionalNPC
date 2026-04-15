from enum import Enum

class Attitude(str, Enum):
    PANICO = "panico / supervivencia"
    DEFENSIVO = "defensivo / cauteloso"
    ALERTA = "alerta / analitico"
    CURIOSO = "curioso / confiado"
    NEUTRAL = "neutral / pasivo"
    SIN_CLASIFICACION = "sin clasificacion"

def get_attitude_from_decision(decision: str) -> str:
    """Mapea una acción difusa hacia un estado de actitud (Attitude Enum)."""
    mapping = {
        "flee": Attitude.PANICO,
        "dormir": Attitude.DEFENSIVO,
        "explore": Attitude.CURIOSO,
        "idle": Attitude.NEUTRAL,
    }
    return mapping.get(decision, Attitude.SIN_CLASIFICACION).value
