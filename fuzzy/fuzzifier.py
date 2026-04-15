# fuzzy/fuzzifier.py

from fuzzy.membership import gaussian


REQUIRED_FUZZY_GENES = [
    "estimulacion_optima_baja", "estimulacion_optima_media", "estimulacion_optima_alta",
    "umbral_bienestar_negativo", "umbral_bienestar_neutral", "umbral_bienestar_positivo",
    "tolerancia_estimulo", "tolerancia_incomodidad",
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
            "low": genome["estimulacion_optima_baja"],
            "medium": genome["estimulacion_optima_media"],
            "high": genome["estimulacion_optima_alta"],
        },
        "valence": {
            "negative": genome["umbral_bienestar_negativo"],
            "neutral": genome["umbral_bienestar_neutral"],
            "positive": genome["umbral_bienestar_positivo"],
        },
        "sigma_arousal": genome["tolerancia_estimulo"],
        "sigma_valence": genome["tolerancia_incomodidad"],
    }

def normalize(memberships):
    total = sum(memberships.values())
    if total == 0:
        return memberships
    return {k: v / total for k, v in memberships.items()}

def fuzzify_arousal(A, genome=None):
    cfg = _resolve_fuzzy_params(genome)
    memberships = {
        "low": gaussian(A, cfg["arousal"]["low"], cfg["sigma_arousal"]),
        "medium": gaussian(A, cfg["arousal"]["medium"], cfg["sigma_arousal"]),
        "high": gaussian(A, cfg["arousal"]["high"], cfg["sigma_arousal"])
    }
    return normalize(memberships)

def fuzzify_valence(V, genome=None):
    cfg = _resolve_fuzzy_params(genome)
    memberships = {
        "negative": gaussian(V, cfg["valence"]["negative"], cfg["sigma_valence"]),
        "neutral": gaussian(V, cfg["valence"]["neutral"], cfg["sigma_valence"]),
        "positive": gaussian(V, cfg["valence"]["positive"], cfg["sigma_valence"])
    }
    return normalize(memberships)
