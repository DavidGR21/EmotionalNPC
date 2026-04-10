def evaluate_rules(Af, Vf):
    rules = {
        "flee": min(Af["high"], Vf["negative"]),
        "hide": min(Af["medium"], Vf["negative"]),
        "idle_neg": min(Af["low"], Vf["negative"]),

        "observe1": min(Af["high"], Vf["neutral"]),
        "observe2": min(Af["medium"], Vf["neutral"]),
        "observe3": min(Af["low"], Vf["neutral"]),

        "explore1": min(Af["high"], Vf["positive"]),
        "explore2": min(Af["medium"], Vf["positive"]),
        "idle_pos": min(Af["low"], Vf["positive"]),
    }

    # Agregación por acción
    actions = {
        "flee": rules["flee"],
        "hide": rules["hide"],
        "observe": max(rules["observe1"], rules["observe2"], rules["observe3"]),
        "explore": max(rules["explore1"], rules["explore2"]),
        "idle": max(rules["idle_neg"], rules["idle_pos"])
    }

    return actions

def normalize_actions(actions):
    total = sum(actions.values())
    if total == 0:
        return actions
    return {k: v / total for k, v in actions.items()}