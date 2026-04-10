def choose_action(actions):
    if not actions:
        return None

    return max(actions, key=actions.get)