from typing import Callable

import numpy as np


class Layer:

    def __init__(
        self,
        num_inputs: int,
        num_neurons: int,
        activation_func: Callable[[float], float],
        derivative_activation: Callable[[float], float],
        weight_range: tuple[float, float] = (-0.2, 0.2),
        bias_range: tuple[float, float] = (-0.2, 0.2),
    ):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.activation_func = activation_func
        self.derivative_activation = derivative_activation
        self.weight_range = weight_range
        self.bias_range = bias_range

        b_min, b_max = bias_range
        w_min, w_max = weight_range

        # Inicializa os Bias de cada neurônio com um valor aleatório
        # em dado intervalo, armazenando em um vetor de valores
        self.bias: list[float] = list(np.random.uniform(
            low=b_min,
            high=b_max,
            size=num_neurons
        ))

        # Inicializa os pesos de cada neurônio com um vetor de tamanho
        # 'número de entradas' com valores aleatórios em dado intervalo,
        # armazenando em um vetor de vetores
        self.weights: list[list[float]] = [
            list(np.random.uniform(
                low=w_min,
                high=w_max,
                size=num_inputs
            ))
            for _ in range(num_neurons)
        ]

        self.last_inputs = []
        self.last_outputs = []

        self.last_delta_weights = [
            [0.0 for _ in range(num_inputs)]
            for _ in range(num_neurons)
        ]
        self.last_delta_bias = [0.0 for _ in range(len(self.weights))]


    # ================

    # ====

    # ====

    # ================

    def forward(self, inputs: list[float]) -> list[float]:
        """
        Executa a propagação direta (Feedforward) da camada.
        """

        if len(inputs) != self.num_inputs:
            raise ValueError(
                "número de entradas deve ser igual "
                "ao número de pesos. "
                f"Entradas: {inputs} | "
                f"Pesos: {self.num_inputs}"
            )

        # Salva entradas para o cálculo de
        # correção de pesos no backward
        self.last_inputs = inputs

        # Itera pelos neurônios, calculando o somatório ponderado + bias e aplicando f(...)
        outputs: list[float] = []
        for i_neuron in range(len(self.weights)):

            # Somatório inicial com bias
            # net_i = b_i + Σ_j (x_j * w_ij)
            y_in_neuron = self.bias[i_neuron]

            for inp, w in zip(inputs, self.weights[i_neuron]):
                y_in_neuron += inp * w

            # Ativação
            # y_i = f(y_in_neuron)
            outputs.append(self.activation_func(y_in_neuron))

        # Salva saídas ativadas para cálculo de f'(y) no backward
        self.last_outputs = outputs

        return outputs

    # ====

    # ====

    def backward(
        self,
        next_layer_deltas: list[float],
        next_layer_weights: list[list[float]],
        learn_rate: float,
        momentum: float = 0.0,
    ) -> list[float]:
        """
        Executa a retropropagação de erro e a atualização de pesos para uma camada escondida.
        """

        current_layer_deltas: list[float] = []

        # Retropropagação do erro da camada seguinte para a atual
        for i_neuron in range(self.num_neurons):

            # δ_in_j = Σ_k (δ_k * w_kj)
            error_sum = 0.0
            for k_next_neuron in range(len(next_layer_deltas)):
                delta_k = next_layer_deltas[k_next_neuron]
                weight_ki = next_layer_weights[k_next_neuron][i_neuron]
                error_sum += delta_k * weight_ki

            # δ_j = δ_in_j * f'(z_j)
            y_i = self.last_outputs[i_neuron]
            delta_i = error_sum * self.derivative_activation(y_i)
            current_layer_deltas.append(delta_i)

        # Correção e atualização de pesos e bias
        for i_neuron in range(self.num_neurons):
            delta_i = current_layer_deltas[i_neuron]

            # Correção do bias com Momentum
            # Δv_0j = α * δ_j + μ * Δv_0j(t-1)
            delta_bias = (learn_rate * delta_i) + (momentum * self.last_delta_bias[i_neuron])

            # Atualização 
            # v_0j = v_0j + Δv_0j
            self.bias[i_neuron] += delta_bias
            self.last_delta_bias[i_neuron] = delta_bias

            # Correção dos pesos com Momentum
            for j_input in range(self.num_inputs):

                # Δv_ij = α * δ_j * x_i + μ * Δv_ij(t-1)
                x_j = self.last_inputs[j_input]
                delta_w = (learn_rate * delta_i * x_j) + (momentum * self.last_delta_weights[i_neuron][j_input])

                # Atualização 
                # v_ij = v_ij + Δv_ij
                self.weights[i_neuron][j_input] += delta_w
                self.last_delta_weights[i_neuron][j_input] = delta_w

        return current_layer_deltas
