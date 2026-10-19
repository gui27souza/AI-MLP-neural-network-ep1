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

        if num_inputs != hidden_layers[0].num_inputs:
            raise ValueError(
                "número de entradas deve ser igual " \
                "o mesmo da primeira camada escondida"
            )

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers

        # Têm o mesmo número de entradas do que o número de
        # saídas da camada anterior, ou seja, da última
        # camada escondida
        self.output_layer = Layer(
            num_inputs=hidden_layers[-1].num_neurons,
            num_neurons=num_outputs,
            activation_func=output_activation,
            derivative_activation=output_deriv_activation,
        )
        self.num_outputs = num_outputs

        self.learn_rate = learn_rate


    # =====


    def run_epoch(
        self,
        inputs_list: list[list[float]],
        targets: list[list[float]]
    ):

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

        return outputs


    # =====


    # TODO - MLP - aprendizado da rede com backpropagation

