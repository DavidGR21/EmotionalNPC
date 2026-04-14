import json
import os


BASE_DIR = os.path.dirname(os.path.dirname(__file__))


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
                return db.get(personality_type, db.get("normal", {}))
            except json.JSONDecodeError:
                pass
    return {}
