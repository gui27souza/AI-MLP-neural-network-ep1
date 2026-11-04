import os

from .models import Layer, MLP
from .funcs import sigmoide_binaria, deriv_sigmoide_binaria
from .orchestrator import Orchestrator


NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 1000))


if __name__ == "__main__":

    hidden_layer = Layer(
        num_inputs=2, num_neurons=1,
        activation_func=sigmoide_binaria,
        derivative_activation=deriv_sigmoide_binaria,
    )

    mlp = MLP(
        num_inputs=2, num_outputs=1,
        learn_rate=0.25,
        hidden_layers=[hidden_layer],
        output_activation=sigmoide_binaria,
        output_deriv_activation=deriv_sigmoide_binaria,
    )

    inputs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]

    targets_AND = [
        [0.0],
        [0.0],
        [0.0],
        [1.0]
    ]

    targets_OR = [
        [0.0],
        [1.0],
        [1.0],
        [1.0]
    ]

    orch_AND = Orchestrator(
        mlp=mlp, epochs=NUM_EPOCHS,
        inputs=inputs, targets=targets_AND,
    )

    orch_OR = Orchestrator(
        mlp=mlp, epochs=NUM_EPOCHS,
        inputs=inputs, targets=targets_OR,
    )

    orchs = [
        orch_AND,
        orch_OR,
    ]

    for orch in orchs:

        print_interval = max(1, int(NUM_EPOCHS / 5))
        orch.run(print_epochs_interval=print_interval)

        print("")
        orch.print_results()
        print("")
