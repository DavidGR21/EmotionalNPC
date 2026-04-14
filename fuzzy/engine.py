from fuzzy.fuzzifier import fuzzify_arousal, fuzzify_valence
from fuzzy.rules import evaluate_rules, normalize_actions
from fuzzy.inference import choose_action
from typing import Tuple, Dict

def process_fuzzy_logic(emotion, genome: dict, include_action_scores: bool = False):
    """
    Patrón Façade: Encapsula todo el pipeline de procesamiento de lógica difusa:
    Fuzificación (Numérico -> Lingüístico) -> Evaluación de Reglas -> Desfuzificación.
    
    Retorna:
    - La decisión ganadora (ej. 'flee').
    - Un diccionario con el 'top 2' de opciones para renderización visual de HUD.
    - Opcional: el diccionario completo de acciones normalizadas si include_action_scores=True.
    """
    
    # 1. Fuzificación (Interprete Subjetivo)
    af = fuzzify_arousal(emotion.Arousal, genome)
    vf = fuzzify_valence(emotion.Valence, genome)
    
    # 2. Base de Conocimiento (Evaluación de Reglas)
    actions = normalize_actions(evaluate_rules(af, vf))
    
    # 3. Decisión final (Desdifusificación)
    decision = choose_action(actions)
    
    # 4. Procesamiento de salida HUD (Opciones conflictivas)
    sorted_actions = sorted(actions.items(), key=lambda x: x[1], reverse=True)
    top_opts = {k: round(v, 4) for k, v in sorted_actions[:2]}
    
    if include_action_scores:
        return decision, top_opts, actions

    return decision, top_opts
