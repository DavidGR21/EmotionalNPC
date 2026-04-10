from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from models.npc import Npc
from controllers.personality_controller import load_personality

app = FastAPI(title="Emotional NPC Engine API")

# Memoria RAM: Diccionario mundial para almacenar la sesión activa de los NPCs instanciados por Godot
active_npcs: Dict[str, Npc] = {}


# ====== ESQUEMAS DE VALIDACIÓN (Contratos de Datos) ======
class NpcInitSchema(BaseModel):
    id: str
    personality: str = "normal"

class EnvironmentSchema(BaseModel):
    sound: float
    threat: float
    light: float


# ====== ENDPOINTS (Rutas HTTP para las peticiones) ======

@app.post("/npc/init")
def init_npc(req: NpcInitSchema):
    """
    Endpoint para dar vida a un nuevo NPC copiando los parámetros de su personalidad elegida.
    """
    try:
        # El controlador se encarga de lidiar con los archivos
        params = load_personality(req.personality)
        if not params:
            raise ValueError(f"No se encontró ADN o weights para la personalidad: '{req.personality}'")
            
        nuevo_ente = Npc(name=req.id, params=params, personality_type=req.personality)
        active_npcs[req.id] = nuevo_ente
        return {"status": "success", "message": f"[{req.personality.upper()}] asignado a '{req.id}'"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/npc/{npc_id}/update")
def update_npc(npc_id: str, env_data: EnvironmentSchema):
    """
    Endpoint para inyectar entorno en tiempo real a la mente del NPC solicitado.
    """
    if npc_id not in active_npcs:
        raise HTTPException(status_code=404, detail=f"NPC '{npc_id}' no encontrado en RAM.")
        
    npc_instance = active_npcs[npc_id]
    
    # Modelo matemático (Puro)
    json_response = npc_instance.step(
        sound=env_data.sound,
        threat=env_data.threat,
        light=env_data.light
    )
    
    return json_response
