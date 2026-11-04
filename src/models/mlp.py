from typing import Callable

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
        )

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        self.learn_rate = learn_rate


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
                print(f"in: {input_data} \t-> out: {output} \t| target: {target}")

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

        # Guarda os pesos originais da camada de saída antes da
        # atualização, pois a última camada escondida precisará
        # deles para calcular seu próprio delta
        next_weights = self.output_layer.weights
        next_deltas = out_deltas

        # Atualiza pesos e bias da camada de saída
        for i_neuron in range(self.output_layer.num_neurons):
            delta_i = out_deltas[i_neuron]
            self.output_layer.bias[i_neuron] += self.learn_rate * delta_i
            for j_input in range(self.output_layer.num_inputs):
                x_j = self.output_layer.last_inputs[j_input]
                self.output_layer.weights[i_neuron][j_input] += (
                    self.learn_rate * delta_i * x_j
                )

        # Propaga o erro pelas camadas escondidas (se houver alguma)
        for _, layer in reversed(list(enumerate(self.hidden_layers))):

            # Salva a matriz de pesos atual antes do backward
            # alterar seus valores
            current_layer_weights = layer.weights

            next_deltas = layer.backward(
                next_layer_deltas=next_deltas,
                next_layer_weights=next_weights,
                learn_rate=self.learn_rate,
            )

            # Os pesos não atualizados desta camada viram a
            # referência da camada seguinte
            next_weights = current_layer_weights
