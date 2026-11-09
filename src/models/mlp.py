from typing import Callable
import json
import os
import numpy as np

from .layer import Layer

# ===


class MLP:

    def __init__(
        self,
        num_inputs: int, num_outputs: int,
        learn_rate: float,
        hidden_layers: list[Layer],
        output_activation: Callable[[float], float],
        output_deriv_activation: Callable[[float], float],
        weight_range: tuple[float, float] = (-0.2, 0.2),
        bias_range: tuple[float, float] = (-0.2, 0.2),
        momentum: float = 0.0,
    ):
        """
        Inicializa a MLP.

        - A 'camada de entrada' não possui pesos ou neurônios (é apenas o vetor bruto de dados).
        - 'hidden_layers' define as camadas intermediárias de processamento. Podem ser 0 (lista vazia).
        - 'output_layer' é mantida como um atributo separado pois possui um tratamento
          especial no Backpropagation (cálculo direto de erro via target vs output).
        """

        # Trata a conexão de entradas da output_layer caso hidden_layers seja vazia
        if hidden_layers:
            if num_inputs != hidden_layers[0].num_inputs:
                raise ValueError(
                    "número de entradas deve ser igual ao mesmo da primeira camada escondida"
                )

            previous_num_outputs = num_inputs
            for i, layer in enumerate(hidden_layers):
                if layer.num_inputs != previous_num_outputs:
                    raise ValueError(
                        f"número de entradas da camada {i + 1} incompatível "
                        "ao número de saídas da camada anterior. "
                        f"Entradas: {layer.num_inputs} | "
                        f"Saídas da camada anterior: {previous_num_outputs}"
                    )
                previous_num_outputs = layer.num_neurons

            output_layer_inputs = hidden_layers[-1].num_neurons

        # Sem camadas escondidas, a camada de saída
        # conecta direto nas entradas da rede
        else:
            output_layer_inputs = num_inputs

        self.output_layer = Layer(
            num_inputs=output_layer_inputs,
            num_neurons=num_outputs,
            activation_func=output_activation,
            derivative_activation=output_deriv_activation,
            weight_range=weight_range,
            bias_range=bias_range,
        )

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        self.learn_rate = learn_rate
        self.momentum = momentum

    # =====

    def predict(self, inputs: list[float]) -> list[float]:
        """
        Executa apenas a propagação direta (feedforward) sem alterar os pesos.
        Útil para inferência e avaliação de métricas.
        """
        if len(inputs) != self.num_inputs:
            raise ValueError(
                f"número de entradas incompatível: esperado {self.num_inputs} e recebido {len(inputs)}"
            )

        current_inputs = inputs
        for layer in self.hidden_layers:
            current_inputs = layer.forward(current_inputs)

        return self.output_layer.forward(current_inputs)

    def evaluate(
        self, inputs_list: list[list[float]], targets: list[list[float]]
    ) -> tuple[float, float]:
        """
        Calcula o Erro Quadrático Médio (MSE) e a Acurácia para um conjunto de dados.
        - Em problemas multiclasse (num_outputs > 1): usa Winner-Takes-All (argmax).
        - Em problemas binários (num_outputs == 1): avalia o sinal (positivo ou negativo).
        """
        outputs = [self.predict(x) for x in inputs_list]

        outputs_arr = np.array(outputs)
        targets_arr = np.array(targets)

        # MSE = média da diferença ao quadrado
        mse = float(np.mean((targets_arr - outputs_arr) ** 2))

        # Acurácia
        if self.num_outputs > 1:
            preds = np.argmax(outputs_arr, axis=1)
            actuals = np.argmax(targets_arr, axis=1)
            acc = float(np.mean(preds == actuals))
        else:
            preds = np.where(outputs_arr >= 0, 1.0, -1.0)
            acc = float(np.mean(preds == targets_arr))

        return mse, acc

    # =====


    def run_epoch(
        self,
        inputs_list: list[list[float]],
        targets: list[list[float]],
        print_info: bool = False,
    ):

        # Valor deve ser o mesmo para permitir a comparação e 
        # cálculo do erro
        if len(inputs_list) != len(targets):
            raise ValueError(
                "número de entradas deve ser igual ao de targets: "
                f"entradas {len(inputs_list)} e "
                f"targets {len(targets)}"
            )

        for input_data, target in zip(inputs_list, targets):
            output = self._run_iteration(input_data, target)
            if print_info:
                print(f"in: {input_data}\nout:    {output}\ntarget: {target}\n")

    # ===

    def _run_iteration(self, inputs: list[float], target: list[float]):

        if len(inputs) != self.num_inputs:
            raise ValueError(
                "número de entradas incompatível: "
                f"esperado {self.num_inputs} e "
                f"recebido {len(inputs)}"
            )

        if len(target) != self.num_outputs:
            raise ValueError(
                "tamanho de target incompatível: "
                f"esperado {self.num_outputs} e "
                f"recebido {len(target)}"
            )

        current_inputs: list[float] = inputs
        current_outputs: list[float] = inputs
        for layer in self.hidden_layers:
            current_outputs = layer.forward(current_inputs)
            current_inputs = current_outputs

        outputs = self.output_layer.forward(current_outputs)

        # Os pesos devem ser atualizados de forma 'online'
        # ou seja, a cada cálculo de output / iteração
        self._update_weights(target, outputs)

        return outputs


    # =====


    def _update_weights(self, target: list[float], outputs: list[float]):

        # A camada de saída tem tratativa especial, como não há uma
        # camada após ela, sua fórmula de cálculo dos deltas é diferente
        final_layer_deltas: list[float] = []
        for t, out in zip(target, outputs):
            final_layer_deltas.append(
                (t - out) * self.output_layer.derivative_activation(out)
            )

        out_deltas = final_layer_deltas

        # Guarda uma cópia dos pesos originais da camada de saída antes da
        # atualização, pois a última camada escondida precisará deles
        # (pesos anteriores à correção) para calcular seu próprio delta
        next_weights = [w[:] for w in self.output_layer.weights]
        next_deltas = out_deltas

        # Atualiza pesos e bias da camada de saída com termo de correção e momentum
        for i_neuron in range(self.output_layer.num_neurons):
            delta_i = out_deltas[i_neuron]

            # Δb_k = α * δ_k + μ * Δb_k(t-1)
            # b_k = b_k + Δb_k
            delta_bias = (self.learn_rate * delta_i) + (
                self.momentum * self.output_layer.last_delta_bias[i_neuron]
            )
            self.output_layer.bias[i_neuron] += delta_bias
            self.output_layer.last_delta_bias[i_neuron] = delta_bias

            # Δw_jk = α * δ_k * z_j + μ * Δw_jk(t-1)
            # w_jk = w_jk + Δw_jk
            for j_input in range(self.output_layer.num_inputs):
                x_j = self.output_layer.last_inputs[j_input]
                delta_w = (self.learn_rate * delta_i * x_j) + (
                    self.momentum
                    * self.output_layer.last_delta_weights[i_neuron][j_input]
                )
                self.output_layer.weights[i_neuron][j_input] += delta_w
                self.output_layer.last_delta_weights[i_neuron][j_input] = delta_w

        # Propaga o erro pelas camadas escondidas (se houver alguma)
        for _, layer in reversed(list(enumerate(self.hidden_layers))):

            # Salva uma cópia da matriz de pesos atual antes do backward
            # alterar seus valores
            current_layer_weights = [w[:] for w in layer.weights]

            next_deltas = layer.backward(
                next_layer_deltas=next_deltas,
                next_layer_weights=next_weights,
                learn_rate=self.learn_rate,
                momentum=self.momentum,
            )

            # Os pesos não atualizados desta camada viram a
            # referência da camada seguinte
            next_weights = current_layer_weights
