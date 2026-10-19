import math

# ===

def sigmoide_binaria(y_in: float) -> float:
    return 1 / (1 + math.exp(-1*y_in))

def deriv_sigmoide_binaria(y: float) -> float:
    return y * (1 - y)

# ===

def sigmoide_bipolar(y_in: float) -> float:
    return (2 / (1 + math.exp(-1*y_in))) - 1

def deriv_sigmoide_bipolar(y: float) -> float:
    return 0.5 * (1 - y) * (1 + y)
