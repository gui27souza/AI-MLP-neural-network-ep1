import math

from src.models import Layer


def sigmoide(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


layer = Layer(
    num_inputs=2, num_neurons=1,
    activation_func=sigmoide,
    derivative_activation=sigmoide,
)

inputs = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
]

outputs: list[list[float]] = []
for inp in inputs:
    outputs.append(layer.forward(inp))

for i, o in zip(inputs, outputs):
    print(f"{i} -> {o}")
