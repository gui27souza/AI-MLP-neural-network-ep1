from typing import Callable

import numpy as np

from src.config import Constants


class Layer:

    def __init__(
        self, num_inputs: int, num_neurons: int,
        activation_func: Callable[[float], float],
        derivative_activation: Callable[[float], float]
    ):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.activation_func = activation_func
        self.derivative_activation = derivative_activation

        # Inicializa os Bias de cada neurônio com um valor aleatório
        # em dado intervalo, armazenando em um vetor de valores
        self.bias: list[float] = list(np.random.uniform(
            low=Constants.MIN_INIT_BIAS,
            high=Constants.MAX_INIT_BIAS,
            size=num_neurons
        ))

        # Inicializa os pesos de cada neurônio com um vetor de tamanho
        # 'número de entradas' com valores aleatórios em dado intervalo,
        # armazenando em um vetor de vetores
        self.weights: list[list[float]] = [
            list(np.random.uniform(
                low=Constants.MIN_INIT_WEIGHTS,
                high=Constants.MAX_INIT_WEIGHTS,
                size=num_inputs
            ))
            for _ in range(num_neurons)
        ]

        self.last_inputs = []
        self.last_outputs = []


    # ================

    # ====

    # ================

    # TODO - Layer.forward
    def forward(self):
        pass

    # TODO - Layer.backward
    def backward(self):
        pass

    # TODO - Layer.update_weights
    def update_weights(self):
        pass
