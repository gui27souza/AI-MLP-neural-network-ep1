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
    ) -> None:

        # Essa checagem funciona na prática pois a camada de
        # entrada não existe de fato, é uma abstração para que
        # façamos o input na primeira camada escondida
        if num_inputs != hidden_layers[0].num_inputs:
            raise ValueError(
                "número de entradas deve ser igual " \
                "o mesmo da primeira camada escondida"
            )

        # Checagem de segurança entre as camadas escondidas,
        # garante que tenham entradas e saídas compatíveis
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

        # Têm o mesmo número de entradas do que o número de
        # saídas da camada anterior, ou seja, da última
        # camada escondida
        self.output_layer = Layer(
            num_inputs=hidden_layers[-1].num_neurons,
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
        targets: list[list[float]]
    ):

        # Valor deve ser o mesmo para permitir a comparação e 
        # cálculo do erro
        if len(inputs_list) != len(targets):
            raise ValueError(
                "número de entradas deve ser igual ao de targets: "
                f"entradas {len(inputs_list)} e "
                f"targets {len(targets)}"
            )

        for input, target in zip(inputs_list, targets):
            output = self._run_iteration(input, target)
            print(f"in: {input} \t-> out: {output} \t| target: {target}")


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


    # TODO - MLP - aprendizado da rede com backpropagation

    def _update_weights(self, target: list[float], outputs: list[float]):

        out_deltas = self._output_layer_deltas(target, outputs)

    def _output_layer_deltas(
        self, target: list[float], outputs: list[float]
    ) -> list[float]:

        deltas: list[float] = []

        for t, out in zip(target, outputs):
            deltas.append(
                (t - out) * self.output_layer.derivative_activation(out)
            )

        return deltas
