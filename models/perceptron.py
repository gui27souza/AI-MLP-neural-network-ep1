from typing import Callable


class Perceptron:

    def __init__(
        self,
        activation_func: Callable[[float], float],
        weights: list[float], bias: float
    ):
        self.activation_func = activation_func
        self.weights = weights
        self.bias = bias

   # ===

    def _weighted_sum(self, inputs: list[float]) -> float:

        if len(inputs) != len(self.weights):
            raise ValueError("the number of inputs must match the number of weights")

        z: float = 0

        for i, w in zip(inputs, self.weights):
            z += i * w

        z += self.bias

        return z


    def calculate_output(self, inputs: list[float]) -> float:

        z = self._weighted_sum(inputs)

        return self.activation_func(z)
