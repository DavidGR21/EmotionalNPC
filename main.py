import uvicorn
# Importamos 'app' para validar que api.py cargue bien, pero la magia
# de uvicorn es que invoca directamente a "api:app" por string.
from api import app

if __name__ == "__main__":
    print("===== MOTOR EMOTIONAL NPC =====")
    print("Levantando servidor y conectando endpoints de la IA...")
    # Delegamos toda la responsabilidad y el ciclo de vida a Uvicorn y a api.py
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
