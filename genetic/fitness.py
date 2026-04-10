from models.emotion import Emotion
from fuzzy.fuzzifier import fuzzify_arousal, fuzzify_valence
from fuzzy.rules import evaluate_rules, normalize_actions
from fuzzy.inference import choose_action

# En vez de importar main, que tiene dependencias globales, copiamos la lógica 
# puramente de evaluación aquí o importamos su timeline generator.
# Asumamos que podemos importarla pero si hay efectos secundarios, mejor proveemos 
# los steps como un array. Vamos a importar la funcion pura:
from main import build_environment_timeline

def evaluate_fitness(genome):
    """
    Simula la vida del NPC con 'genome' (que son los `params`).
    Devuelve un float (score) más alto si el NPC es más apto
    para sobrevivir y explorar sin sufrir colapsos de estrés.
    """
    
    # Creamos un sistema emocional nuevo aislado para esta evaluación
    emotion = Emotion()
    
    # Obtenemos la prueba de estres
    timeline = build_environment_timeline()
    
    score = 0.0
    
    for tick, (phase, env) in enumerate(timeline):
        emotion.update(env, genome)
        
        af = fuzzify_arousal(emotion.Arousal)
        vf = fuzzify_valence(emotion.Valence)
        
        actions = normalize_actions(evaluate_rules(af, vf))
        decision = choose_action(actions)
        
        # --- DEFINICIÓN DE FITNESS (Personalidad "Normal" Realista) ---
        
        # Premiar que la emoción de estrés sea coherente con la amenaza externa
        abs_error_stress = abs(emotion.stress - env.threat)
        score -= abs_error_stress * 2.0  # El estrés debería seguir intuitivamente a la amenaza
        
        # Recompensas contextuales según la fase del entorno
        if phase == "calm":
            if decision in ["idle", "observe"]: score += 5
            elif decision in ["flee", "hide"]: score -= 10 # No debe huir si está en calma
            
        elif phase == "escalation":
            if decision in ["observe", "hide"]: score += 5
            elif decision == "explore": score -= 2
            
        elif phase == "peak":
            if decision in ["flee", "hide"]: score += 10
            elif decision in ["idle", "explore"]: score -= 20 # Peligro mortal
            
        elif phase == "recovery":
            if decision in ["hide", "observe"]: score += 5
            
        elif phase == "curiosity":
            if decision == "explore": score += 10
            elif decision == "flee": score -= 10 # Ya no hay amenaza
            
    return score
	