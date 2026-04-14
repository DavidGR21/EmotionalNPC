# fuzzy/fuzzifier.py

from fuzzy.membership import gaussian


REQUIRED_FUZZY_GENES = [
    "cA_low", "cA_medium", "cA_high",
    "cV_negative", "cV_neutral", "cV_positive",
    "sigma",
]


def _resolve_fuzzy_params(genome=None):
    if genome is None:
        raise ValueError("Se requiere genoma entrenado para fuzzificacion.")

    missing = [key for key in REQUIRED_FUZZY_GENES if key not in genome]
    if missing:
        missing_txt = ", ".join(missing)
        raise ValueError(f"Genoma incompleto: faltan genes difusos [{missing_txt}].")

    return {
        "arousal": {
            "low": genome["cA_low"],
            "medium": genome["cA_medium"],
            "high": genome["cA_high"],
        },
        "valence": {
            "negative": genome["cV_negative"],
            "neutral": genome["cV_neutral"],
            "positive": genome["cV_positive"],
        },
        "sigma": genome["sigma"],
    }

def normalize(memberships):
    total = sum(memberships.values())
    if total == 0:
        return memberships
    return {k: v / total for k, v in memberships.items()}

def fuzzify_arousal(A, genome=None):
    cfg = _resolve_fuzzy_params(genome)
    memberships = {
        "low": gaussian(A, cfg["arousal"]["low"], cfg["sigma"]),
        "medium": gaussian(A, cfg["arousal"]["medium"], cfg["sigma"]),
        "high": gaussian(A, cfg["arousal"]["high"], cfg["sigma"])
    }
    return normalize(memberships)

def fuzzify_valence(V, genome=None):
    cfg = _resolve_fuzzy_params(genome)
    memberships = {
        "negative": gaussian(V, cfg["valence"]["negative"], cfg["sigma"]),
        "neutral": gaussian(V, cfg["valence"]["neutral"], cfg["sigma"]),
        "positive": gaussian(V, cfg["valence"]["positive"], cfg["sigma"])
    }
    return normalize(memberships)
