import json
import os


BASE_DIR = os.path.dirname(os.path.dirname(__file__))

LEGACY_TO_RUSSELL = {
    "normal": "calmada",
    "explorador": "feliz",
    "cobarde": "miedosa",
    "valiente": "feliz",
    "superviviente": "triste",
}


def load_personality(personality_type: str) -> dict:
    """
    Controlador para I/O: Lee la BD local y devuelve 
    los pesos numéricos para inyectar en el cerebro del modelo.
    """
    json_path = os.path.join(BASE_DIR, "data", "personalities.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                db = json.load(f)
                normalized = personality_type.lower()
                mapped = LEGACY_TO_RUSSELL.get(normalized, normalized)
                return db.get(mapped, db.get("calmada", {}))
            except json.JSONDecodeError:
                pass
    return {}
