"""
Paso 1 del sistema difuso: Fuzzificacion.

Convierte entradas numericas continuas (Arousal y Valence) en grados de
pertenencia linguisticos usando funciones gaussianas y parametros del genoma.
"""

from fuzzy.membership import gaussian


REQUIRED_FUZZY_GENES = [
    "estimulacion_optima_baja", "estimulacion_optima_media", "estimulacion_optima_alta",
    "umbral_bienestar_negativo", "umbral_bienestar_neutral", "umbral_bienestar_positivo",
    "tolerancia_estimulo", "tolerancia_incomodidad",
]


def _resolve_fuzzy_params(genome=None):
    """Resuelve centros y sigmas difusos desde el genoma entrenado."""
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


def normalize_memberships(memberships):
    """Normaliza membresias para que la suma sea 1 cuando exista activacion."""
    total = sum(memberships.values())
    if total == 0:
        return memberships
    return {k: v / total for k, v in memberships.items()}


def fuzzify_arousal(arousal, genome=None):
    """
    Fuzzifica Arousal en etiquetas low/medium/high.

    Usa los centros de estimulacion optima y sigma de tolerancia al estimulo.
    """
    cfg = _resolve_fuzzy_params(genome)
    memberships = {
        "low": gaussian(arousal, cfg["arousal"]["low"], cfg["sigma_arousal"]),
        "medium": gaussian(arousal, cfg["arousal"]["medium"], cfg["sigma_arousal"]),
        "high": gaussian(arousal, cfg["arousal"]["high"], cfg["sigma_arousal"]),
    }
    return normalize_memberships(memberships)


def fuzzify_valence(valence, genome=None):
    """
    Fuzzifica Valence en etiquetas negative/neutral/positive.

    Usa los umbrales de bienestar y sigma de tolerancia a la incomodidad.
    """
    cfg = _resolve_fuzzy_params(genome)
    memberships = {
        "negative": gaussian(valence, cfg["valence"]["negative"], cfg["sigma_valence"]),
        "neutral": gaussian(valence, cfg["valence"]["neutral"], cfg["sigma_valence"]),
        "positive": gaussian(valence, cfg["valence"]["positive"], cfg["sigma_valence"]),
    }
    return normalize_memberships(memberships)
