import math

def gaussian(x, c, sigma):
    return math.exp(-((x - c) ** 2) / (2 * sigma ** 2))