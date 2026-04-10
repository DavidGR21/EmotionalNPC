# utils/math_utils.py

def clamp(x, min_value=0.0, max_value=1.0):
    return max(min_value, min(x, max_value))