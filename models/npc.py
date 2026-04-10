from models.emotion import Emotion
from models.environment import Environment
from fuzzy.engine import process_fuzzy_logic
from models.enums import get_attitude_from_decision

class Npc:
    def __init__(self, name: str, params: dict, personality_type: str = "normal"):
        self.name = name
        self.personality_type = personality_type
        # El ADN neuronal (pesos genéticos para este NPC)
        self.params = params
        # El cerebro límbico continuo, arranca en estados nulos/medios
        self.emotion = Emotion() 

    def step(self, sound: float, threat: float, light: float) -> dict:
        """
        Inyecta la información sensorial proveniente del cliente (Godot),
        calcula la cascada neurológica y lógica difusa, y retorna el output limpio.
        """
        env = Environment(sound=sound, threat=threat, light=light)
        
        # 1. Update de la matemática continua (El tiempo avanza, el corazón reacciona)
        self.emotion.update(env, self.params)
        
        # 2. Pipeline Mental Desacoplado (Motor de Lógica Difusa)
        decision, top_opts = process_fuzzy_logic(self.emotion)
        
        # 3. Postprocesamiento Estético 
        attitude = get_attitude_from_decision(decision)
        
        # 4. Formatear la respuesta (HUD para Godot)
        return {
            "name": self.name,
            "personality": self.personality_type,
            "metrics": {
                "stress": round(self.emotion.stress, 3),
                "arousal": round(self.emotion.Arousal, 3),
                "valence": round(self.emotion.Valence, 3)
            },
            "decision": decision,
            "attitude": attitude,
            "top_options": top_opts
        }