import os

from src.core.funcs import sigmoide_bipolar, deriv_sigmoide_bipolar
from src.core.models import Layer, MLP
from src.training.orchestrator import Orchestrator


NUM_EPOCHS = int(os.environ["NUM_EPOCHS"])


if __name__ == "__main__":

    inputs = [
        [-1.0, -1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [1.0, 1.0],
    ]

    # =====

    targets_AND = [
        [-1.0],
        [-1.0],
        [-1.0],
        [1.0]
    ]

    mlp_AND = MLP(
        num_inputs=2, num_outputs=1,
        learn_rate=0.25,
        hidden_layers=[],
        output_activation=sigmoide_bipolar,
        output_deriv_activation=deriv_sigmoide_bipolar,
    )

    orch_AND = Orchestrator(
        mlp=mlp_AND, epochs=NUM_EPOCHS,
        inputs=inputs, targets=targets_AND,
    )

    # =====

    targets_OR = [
        [-1.0],
        [1.0],
        [1.0],
        [1.0]
    ]

    mlp_OR = MLP(
        num_inputs=2, num_outputs=1,
        learn_rate=0.25,
        hidden_layers=[],
        output_activation=sigmoide_bipolar,
        output_deriv_activation=deriv_sigmoide_bipolar,
    )

    orch_OR = Orchestrator(
        mlp=mlp_OR, epochs=NUM_EPOCHS,
        inputs=inputs, targets=targets_OR,
    )

    # =====

    targets_XOR = [
        [-1.0],
        [1.0],
        [1.0],
        [-1.0]
    ]

    hidden_layer_1_XOR = Layer(
        num_inputs=2, num_neurons=2,
        activation_func=sigmoide_bipolar,
        derivative_activation=deriv_sigmoide_bipolar,
        weight_range=(-1.0, 1.0),
        bias_range=(-1.0, 1.0),
    )

    mlp_XOR = MLP(
        num_inputs=2, num_outputs=1,
        learn_rate=0.25,
        hidden_layers=[hidden_layer_1_XOR],
        output_activation=sigmoide_bipolar,
        output_deriv_activation=deriv_sigmoide_bipolar,
        weight_range=(-1.0, 1.0),
        bias_range=(-1.0, 1.0),
    )

    orch_XOR = Orchestrator(
        mlp=mlp_XOR, epochs=NUM_EPOCHS,
        inputs=inputs, targets=targets_XOR,
    )

    orchs = [
        orch_AND,
        orch_OR,
        orch_XOR,
    ]

    # =====

    for orch in orchs:

        print_interval = max(1, int(NUM_EPOCHS / 2))
        orch.run(print_epochs_interval=print_interval)

        print("")
        orch.print_results()
        print("")
