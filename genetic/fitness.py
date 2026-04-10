from models.emotion import Emotion
from fuzzy.fuzzifier import fuzzify_arousal, fuzzify_valence
from fuzzy.rules import evaluate_rules, normalize_actions
from fuzzy.inference import choose_action
from main import build_environment_timeline

# ==========================================
# EVALUADOR DEL DESEMPEÑO (FITNESS FUNCTION)
# ==========================================
# Aquí es donde las matemáticas juzgan si un NPC sobrevivirá a la selección natural.
# Usamos el Patrón de Diseño "Factory" (Fábrica): Dependiendo de qué "palabra" le
# pasemos a la función (ej: "cobarde"), esta nos construirá y devolverá una min-función
# de evaluación que premia cosas distintas.

def get_fitness_evaluator(personality_type="normal"):
    """
    Factory que retorna la función de evaluación (fitness) correspondiente
    a la personalidad que queremos generar.
    """
    
    def evaluate_timeline(genome, scoring_logic):
        """
        Wrapper base o "Simulador Universal".
        En lugar de repetir el bucle de la línea del tiempo en cada personalidad,
        esta función corre el simulador base usando los genes recibidos (genome) y 
        simplemente le pregunta a la función `scoring_logic` de abajo cuántos puntos
        se merece en cada instante (tick) simulado.
        """
        # 1. Instanciamos un cerebro emocional limpio exclusivo para este NPC
        emotion = Emotion()
        # 2. Obtenemos el entorno hostil de prueba (el laberinto)
        timeline = build_environment_timeline()
        score = 0.0
        
        # 3. Soltamos al NPC a la arena y simulamos el paso del tiempo
        for tick, (phase, env) in enumerate(timeline):
            # El NPC actualiza sus emociones basado en sus "genes"
            emotion.update(env, genome)
            
            # LÓGICA DIFUSA: Transforma sus niveles a palabras (mucho/poco)
            af = fuzzify_arousal(emotion.Arousal)
            vf = fuzzify_valence(emotion.Valence)
            
            # LÓGICA DIFUSA: Decide qué hacer usando la base de reglas
            actions = normalize_actions(evaluate_rules(af, vf))
            decision = choose_action(actions)
            
            # 4. PASO CRITICO: Le preguntamos al juez qué opina de esa decisión.
            score += scoring_logic(phase, env, emotion, decision)
            
        return score

    # === DEFINICIÓN DE PERFILES PSICOLÓGICOS (SCORING LOGICS / JUECES) ===
    # Cada una de estas funciones es un juez con diferentes gustos. Retornan los "Puntos"
    # que ganará o perderá el individuo en el segundo actual de la simulación temporal.

    def _score_normal(phase, env, emotion, decision):
        """El Juez Normal: Busca coherencia realista paso a paso."""
        delta = 0.0
        # Congruencia básica: El estrés debe ir atado al peligro real
        delta -= abs(emotion.stress - env.threat) * 2.0
        
        if phase == "calm":                 # Si no hay peligro...
            if decision in ["idle", "observe"]: delta += 5
            elif decision in ["flee", "hide"]: delta -= 10 # Se castiga la paranoia
        elif phase == "escalation":         # Si las cosas se calientan...
            if decision in ["observe", "hide"]: delta += 5
            elif decision == "explore": delta -= 2
        elif phase == "peak":               # Ante peligro de muerte inminente...
            if decision in ["flee", "hide"]: delta += 10 # Supervivencia máxima
            elif decision in ["idle", "explore"]: delta -= 20
        elif phase == "recovery":           # Tras escapar...
            if decision in ["hide", "observe"]: delta += 5
        elif phase == "curiosity":          # Si el lugar está bien iluminado y seguro...
            if decision == "explore": delta += 10
            elif decision == "flee": delta -= 10
        return delta

    
    def _score_valiente(phase, env, emotion, decision):
        """El Juez Temerario: Te castiga por huir a menos que literalmente vayas a morir."""
        delta = 0.0
        # Rechaza fuertemente el status cowardice (cobardía)
        if decision == "flee":
            # Un valiente de verdad solo huye si el estrés fisiológico está al limite límite (0.95)
            if emotion.stress > 0.95: delta += 10 
            else: delta -= 30 
        
        if phase in ["peak", "escalation"]:
            # Frente a la amenaza severa, prefiere observar analíticamente en lugar de esconderse ciegamente
            if decision == "observe": delta += 10
            elif decision == "idle": delta += 2
        else:
            if decision == "explore": delta += 5
            
        return delta

    def _score_explorador(phase, env, emotion, decision):
        """El Juez Curioso: Te premia si saltas a la aventura incluso estando asustado, excepto si ves la amenaza en tu cara."""
        delta = 0.0
        
        # Condición Especial (La curiosidad mató al gato): 
        # Alto estrés (tal vez por el silencio aterrador o la oscuridad), PERO no hay amenaza real evidente (<= 0.2).
        if emotion.stress > 0.6 and env.threat <= 0.2:
            if decision == "explore": delta += 50  # Premio MASIVO (se la juega, no le importa el miedo y sale)
            elif decision == "hide": delta -= 10
        else:
            # Si sí hay una amenaza letal latente evidente frente a él -> Comportamiento normal esconderse
            if env.threat > 0.5:
                if decision == "hide": delta += 10
                elif decision == "explore": delta -= 20  # Ser explorador inteligente, no un suicida
            else:
                # Regla general del explorador para el resto de fases
                if decision == "explore": delta += 10
                elif decision == "idle": delta -= 2
        return delta
        
    def _score_cobarde(phase, env, emotion, decision):
        """El Juez Paranoico: Te premia si huyes de tu propia sombra."""
        delta = 0.0
        # Reacciona exageradamente a todo lo que suba el stress
        if env.threat > 0.1 or env.sound > 0.3:
            # Recompensa alta por esconderse ante el mínimo sonido
            if decision in ["hide", "flee"]: delta += 10
            elif decision == "explore": delta -= 20
        else:
            # Recompensa incluso estando a salvo si decide esconderse
            if decision == "hide": delta += 2 
            elif decision == "explore": delta -= 5
            
        # Pánico absoluto: Si su corazón late rápido, huir le da puntos extras
        if emotion.stress >= 0.8 and decision == "flee":
            delta += 15
            
        return delta    
    def _score_superviviente(phase, env, emotion, decision):
        """El Juez Superviviente: Táctico e implacable. Prioriza salvaguardar la vida, pero no es paralizado por el miedo."""
        delta = 0.0
        
        # Supervivencia al límite: Penalización si el estrés llega a niveles fatales y no hace nada consecuente.
        if emotion.stress > 0.9:
            if decision not in ["hide", "flee"]: delta -= 50 # Riesgo de muerte por pasividad
            else: delta += 15 # Reacción biológica correcta de salvamento
            
        # Reacción objetiva a las amenazas sin depender de qué fase estemos arbitrariamente:
        if env.threat > 0.7:
            # Peligro alto activo
            if decision in ["flee", "hide"]: delta += 10
            else: delta -= 30
        elif env.threat > 0.2:
            # Peligro latente o subiendo
            if decision == "observe": delta += 10 # Calcula sus opciones
            elif decision == "explore": delta -= 15 # Demasiado riesgo
        else:
            # Entorno calmado
            if decision == "explore": delta += 10 # Busca recursos (scavenging) oportunistamente
            elif decision == "idle": delta -= 5   # No pierde su tiempo quedándose parado
            
        return delta
        
    # ==========================================
    # EL MOTOR DE LA FÁBRICA (INYECCION DEL JUEZ)
    # ==========================================
    # Retornamos una función lambda "Lista para usar" (Closure). 
    # Cuando train_ga.py llame a esta lambda con sus genes `genome`,
    # ejecutará nuestro simulador pasándole el juez específico que fue pedido.
    if personality_type == "cobarde":
        return lambda genome: evaluate_timeline(genome, _score_cobarde)
    elif personality_type == "valiente":
        return lambda genome: evaluate_timeline(genome, _score_valiente)
    elif personality_type == "explorador":
        return lambda genome: evaluate_timeline(genome, _score_explorador)
    elif personality_type == "superviviente":
        return lambda genome: evaluate_timeline(genome, _score_superviviente)
    else:
        return lambda genome: evaluate_timeline(genome, _score_normal)