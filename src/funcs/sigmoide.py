import math

# ===

def sigmoide_binaria(y_in: float) -> float:
    """
    Função de ativação sigmoide logística binária (intervalo [0, 1]).
    """
    y_in = max(-500.0, min(500.0, y_in))
    return 1 / (1 + math.exp(-1 * y_in))


def deriv_sigmoide_binaria(y: float) -> float:
    """
    Derivada da sigmoide binária expressa em termos do valor ativado y = f(y_in).
    """
    return y * (1 - y)

# ===

def sigmoide_bipolar(y_in: float) -> float:
    """
    Função de ativação sigmoide logística bipolar / tangente hiperbólica escalada (intervalo [-1, 1]).
    """
    y_in = max(-500.0, min(500.0, y_in))
    return (2 / (1 + math.exp(-1 * y_in))) - 1


def deriv_sigmoide_bipolar(y: float) -> float:
    """
    Derivada da sigmoide bipolar expressa em termos do valor ativado y = f(y_in).
    """
    return 0.5 * (1 - y) * (1 + y)
