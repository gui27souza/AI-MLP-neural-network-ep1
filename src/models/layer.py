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

    # ====

    # ================

    def forward(self, inputs: list[float]):

        if len(inputs) != self.num_inputs:
            raise ValueError(
                "número de entradas deve ser igual "
                "ao número de pesos. "
                f"Entradas: {inputs} | "
                f"Pesos: {self.num_inputs}"
            )

        # Salva entradas para ajuste nos passos seguintes
        self.last_inputs = inputs

        # Itera pelos neurônios, calculando o y_in de cada e armazenando
        # no vetor de resultados após aplicar a função de ativação
        outputs: list[float] = []
        for i_neuron in range(len(self.weights)):
            y_in_neuron = self.bias[i_neuron]
            for inp, w in zip(inputs, self.weights[i_neuron]):
                y_in_neuron += inp * w
            outputs.append(self.activation_func(y_in_neuron))

        # Salva resultados para ajuste nos passos seguintes
        self.last_outputs = outputs

        return outputs

    # ====

    # ====

    def backward(
        self,
        next_layer_deltas: list[float],
        next_layer_weights: list[list[float]],
        learn_rate: float,
    ) -> list[float]:
        """
        Calcula os deltas da camada atual usando os deltas e pesos da camada seguinte,
        atualiza os pesos e biases locais e retorna os novos deltas para a camada anterior.
        """

        current_layer_deltas: list[float] = []

        # Cálculo dos novos deltas
        for i_neuron in range(self.num_neurons):

            error_sum = 0.0
            for k_next_neuron in range(len(next_layer_deltas)):
                delta_k = next_layer_deltas[k_next_neuron]
                weight_ki = next_layer_weights[k_next_neuron][i_neuron]
                error_sum += delta_k * weight_ki

            y_i = self.last_outputs[i_neuron]
            delta_i = error_sum * self.derivative_activation(y_i)
            current_layer_deltas.append(delta_i)

        # Atualização de pesos e bias com o novo delta
        for i_neuron in range(self.num_neurons):
            delta_i = current_layer_deltas[i_neuron]

            self.bias[i_neuron] += learn_rate * delta_i

            for j_input in range(self.num_inputs):
                x_j = self.last_inputs[j_input]
                self.weights[i_neuron][j_input] += learn_rate * delta_i * x_j

        return current_layer_deltas
