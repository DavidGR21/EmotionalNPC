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
        # Atributos fisiológicos/psicológicos explícitos para trazabilidad del modelo.
        self.recepcion_sonoro = params.get("recepcion_sonoro", 0.0)
        self.recepcion_amenaza = params.get("recepcion_amenaza", 0.0)
        self.recepcion_nivel_luz = params.get("recepcion_nivel_luz", 0.0)
        self.recepcion_estres = params.get("recepcion_estres", 0.0)
        self.comfort_luz = params.get("comfort_luz", 0.0)
        self.vulnerabilidad_peligro = params.get("vulnerabilidad_peligro", 0.0)
        self.misofonia = params.get("misofonia", 0.0)
        self.vulnerabilidad_estres = params.get("vulnerabilidad_estres", 0.0)
        self.sensibilidad_amenaza = params.get("sensibilidad_amenaza", 0.05)
        self.sensibilidad_auditiva = params.get("sensibilidad_auditiva", 0.05)
        self.sensibilidad_oscuridad = params.get("sensibilidad_oscuridad", 0.05)
        self.volatilidad = params.get("volatilidad", 0.05)
        self.estabilidad_emocional = params.get("estabilidad_emocional", 0.05)
        self.resiliencia_estres = params.get("resiliencia_estres", 0.05)
        self.estimulacion_optima_baja = params.get("estimulacion_optima_baja", 0.0)
        self.estimulacion_optima_media = params.get("estimulacion_optima_media", 0.5)
        self.estimulacion_optima_alta = params.get("estimulacion_optima_alta", 1.0)
        self.umbral_bienestar_negativo = params.get("umbral_bienestar_negativo", 0.0)
        self.umbral_bienestar_neutral = params.get("umbral_bienestar_neutral", 0.5)
        self.umbral_bienestar_positivo = params.get("umbral_bienestar_positivo", 1.0)
        self.tolerancia_estimulo = params.get("tolerancia_estimulo", 0.15)
        self.tolerancia_incomodidad = params.get("tolerancia_incomodidad", 0.15)
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
        decision, top_opts = process_fuzzy_logic(self.emotion, self.params)
        
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