# fuzzy/fuzzifier.py

from fuzzy.membership import gaussian

#Se optimice con AG
CENTERS = {
    "low": 0.1,
    "medium": 0.5,
    "high": 0.9
}

SIGMA = 0.15

def normalize(memberships):
    total = sum(memberships.values())
    if total == 0:
        return memberships
    return {k: v / total for k, v in memberships.items()}

def fuzzify_arousal(A):
    memberships = {
        "low": gaussian(A, CENTERS["low"], SIGMA),
        "medium": gaussian(A, CENTERS["medium"], SIGMA),
        "high": gaussian(A, CENTERS["high"], SIGMA)
    }
    return normalize(memberships)

def fuzzify_valence(V):
    memberships = {
        "negative": gaussian(V, CENTERS["low"], SIGMA),
        "neutral": gaussian(V, CENTERS["medium"], SIGMA),
        "positive": gaussian(V, CENTERS["high"], SIGMA)
    }
    return normalize(memberships)
