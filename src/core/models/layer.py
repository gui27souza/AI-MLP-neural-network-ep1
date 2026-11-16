from typing import Callable

import numpy as np


class Layer:
    """
    Representa uma camada neural densa (Camada Escondida ou Camada de Saída).

    Nomenclatura teórica (Fausett, Seção 6.1.2 / Material de Aula):
    - Camada Escondida (p neurônios, n entradas):
      * Pesos: matriz V de dimensão (p x n) e vetor de bias v_0j de dimensão (p).
      * Potencial de ativação: z_in_j = v_0j + Σ_i (x_i * v_ij)
      * Saída ativada: z_j = f(z_in_j)
    - Camada de Saída (m neurônios, p entradas):
      * Pesos: matriz W de dimensão (m x p) e vetor de bias w_0k de dimensão (m).
      * Potencial de ativação: y_in_k = w_0k + Σ_j (z_j * w_jk)
      * Saída ativada: y_k = f(y_in_k)
    """

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

        # Inicializa os biases (v_0j ou w_0k) de cada neurônio no intervalo especificado
        self.bias: list[float] = list(
            np.random.uniform(low=b_min, high=b_max, size=num_neurons)
        )

        # Inicializa a matriz de pesos (v_ij ou w_jk) de dimensão (num_neurons x num_inputs)
        self.weights: list[list[float]] = [
            list(np.random.uniform(low=w_min, high=w_max, size=num_inputs))
            for _ in range(num_neurons)
        ]

        # Estado em cache da última propagação direta (usado no Backpropagation)
        self.last_inputs: list[float] = []
        self.last_outputs: list[float] = []

        # Acumuladores da variação anterior de pesos (Δv(t-1) ou Δw(t-1)) para o Momentum
        self.last_delta_weights: list[list[float]] = [
            [0.0 for _ in range(num_inputs)] for _ in range(num_neurons)
        ]
        self.last_delta_bias: list[float] = [0.0 for _ in range(num_neurons)]

    # =========================================================================
    #  PROPAGAÇÃO DIRETA (FEEDFORWARD — Slide 3)
    # =========================================================================

    def compute_net_input(self, i_neuron: int, inputs: list[float]) -> float:
        """
        Calcula o potencial de ativação (combinação linear + bias) de um neurônio:
        z_in_j = v_0j + Σ_i (x_i * v_ij)   [ou y_in_k = w_0k + Σ_j (z_j * w_jk)]
        """
        net_in = self.bias[i_neuron]
        for inp, weight in zip(inputs, self.weights[i_neuron]):
            net_in += inp * weight
        return net_in

    def forward(self, inputs: list[float]) -> list[float]:
        """
        Executa a fase de Feedforward da camada:
        Para cada neurônio, calcula o somatório ponderado (net_in) e aplica a ativação f(net_in).
        """
        if len(inputs) != self.num_inputs:
            raise ValueError(
                "Número de entradas deve ser igual ao número de pesos. "
                f"Entradas: {len(inputs)} | Pesos esperados: {self.num_inputs}"
            )

        # Armazena as entradas (x_i ou z_j) para uso na atualização de pesos do Backpropagation
        self.last_inputs = inputs

        outputs: list[float] = []
        for i_neuron in range(self.num_neurons):
            y_in_neuron = self.compute_net_input(i_neuron, inputs)
            outputs.append(self.activation_func(y_in_neuron))

        # Armazena as saídas ativadas (z_j ou y_k) para o cálculo da derivada f'(net_in)
        self.last_outputs = outputs
        return outputs

    # =========================================================================
    #  RETROPROPAGAÇÃO DO ERRO (BACKPROPAGATION — Slides 4 e 5)
    # =========================================================================

    def compute_output_deltas(
        self, target: list[float], outputs: list[float]
    ) -> list[float]:
        """
        Calcula o termo de informação de erro para a camada de saída (Passo 6 de Fausett — Slide 4):
        δ_k = (t_k - y_k) * f'(y_in_k)
        """
        output_deltas: list[float] = []
        for t_k, y_k in zip(target, outputs):
            delta_k = (t_k - y_k) * self.derivative_activation(y_k)
            output_deltas.append(delta_k)
        return output_deltas

    def compute_hidden_deltas(
        self,
        next_layer_deltas: list[float],
        next_layer_weights: list[list[float]],
    ) -> list[float]:
        """
        Calcula a contribuição do erro na camada escondida (Passo 7 de Fausett — Slide 5):
        1. Soma ponderada dos erros da camada seguinte: δ_in_j = Σ_k (δ_k * w_jk)
        2. Multiplicação pela derivada da ativação:     δ_j = δ_in_j * f'(z_in_j)
        """
        hidden_deltas: list[float] = []
        for j_neuron in range(self.num_neurons):
            # δ_in_j = Σ_k (δ_k * w_jk)
            delta_in_j = 0.0
            for k_next, delta_k in enumerate(next_layer_deltas):
                weight_jk = next_layer_weights[k_next][j_neuron]
                delta_in_j += delta_k * weight_jk

            # δ_j = δ_in_j * f'(z_in_j)
            z_j = self.last_outputs[j_neuron]
            delta_j = delta_in_j * self.derivative_activation(z_j)
            hidden_deltas.append(delta_j)

        return hidden_deltas

    # =========================================================================
    #  ATUALIZAÇÃO DE PESOS SINÁPTICOS (REGRA DELTA — Slide 6)
    # =========================================================================

    def update_weights(
        self,
        deltas: list[float],
        learn_rate: float,
        momentum: float = 0.0,
    ) -> None:
        """
        Atualiza os pesos sinápticos e biases da camada via Gradiente Descendente
        (Passo 8 de Fausett — Slide 6), com suporte opcional a termo de Momentum (μ):
        - Bias:  Δw_0 = α * δ + μ * Δw_0(t-1)       =>  w_0 <- w_0 + Δw_0
        - Pesos: Δw_ji = α * δ * x_i + μ * Δw_ji(t-1) =>  w_ji <- w_ji + Δw_ji
        """
        for i_neuron in range(self.num_neurons):
            delta_i = deltas[i_neuron]

            # Correção e atualização do bias
            delta_bias = (learn_rate * delta_i) + (
                momentum * self.last_delta_bias[i_neuron]
            )
            self.bias[i_neuron] += delta_bias
            self.last_delta_bias[i_neuron] = delta_bias

            # Correção e atualização dos pesos conectados às entradas da camada
            for j_input in range(self.num_inputs):
                inp_val = self.last_inputs[j_input]
                delta_w = (learn_rate * delta_i * inp_val) + (
                    momentum * self.last_delta_weights[i_neuron][j_input]
                )
                self.weights[i_neuron][j_input] += delta_w
                self.last_delta_weights[i_neuron][j_input] = delta_w

    def backward(
        self,
        next_layer_deltas: list[float],
        next_layer_weights: list[list[float]],
        learn_rate: float,
        momentum: float = 0.0,
    ) -> list[float]:
        """
        Calcula os deltas da camada escondida e aplica imediatamente a atualização de pesos.
        """
        current_layer_deltas = self.compute_hidden_deltas(
            next_layer_deltas=next_layer_deltas,
            next_layer_weights=next_layer_weights,
        )
        self.update_weights(
            deltas=current_layer_deltas,
            learn_rate=learn_rate,
            momentum=momentum,
        )
        return current_layer_deltas

    # =========================================================================
    #  GERENCIAMENTO DE ESTADO E CHECKPOINTS
    # =========================================================================

    def reset_deltas(self) -> None:
        """Zera os acumuladores de variação anterior de pesos e bias (momentum)."""
        self.last_delta_weights = [
            [0.0 for _ in range(self.num_inputs)]
            for _ in range(self.num_neurons)
        ]
        self.last_delta_bias = [0.0 for _ in range(self.num_neurons)]

    def get_weights(self) -> dict:
        """Retorna uma cópia profunda dos pesos e biases da camada."""
        return {
            "bias": self.bias[:],
            "weights": [w[:] for w in self.weights],
        }

    def set_weights(self, state: dict) -> None:
        """Restaura os pesos e biases da camada a partir de um snapshot e reseta o momentum."""
        self.bias = state["bias"][:]
        self.weights = [w[:] for w in state["weights"]]
        self.reset_deltas()
