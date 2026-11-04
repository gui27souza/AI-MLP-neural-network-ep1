import os

from src.dataset_loader import DatasetLoader

from src.funcs import (
    sigmoide_bipolar, deriv_sigmoide_bipolar,
)
from src.models import (
    Layer, MLP,
)
from src.orchestrator import Orchestrator
from src.mlp_persister import MLPPersister


NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 1000))


if __name__ == "__main__":

    dataset_path = "data/caracteres-Fausett/caracteres-limpo.csv"

    inputs, targets = DatasetLoader.load_csv(
        filepath=dataset_path,
        num_target_columns=7,
    )

    layer = Layer(
        num_inputs=len(inputs[0]),
        num_neurons=32,
        activation_func=sigmoide_bipolar,
        derivative_activation=deriv_sigmoide_bipolar,
    )

    mlp = MLP(
        num_inputs=len(inputs[0]),
        num_outputs=len(targets[0]),
        learn_rate=0.1,
        hidden_layers=[layer],
        output_activation=sigmoide_bipolar,
        output_deriv_activation=deriv_sigmoide_bipolar,
    )

    orch = Orchestrator(
        mlp=mlp, epochs=NUM_EPOCHS, inputs=inputs, targets=targets
    )

    orch.run(
        print_epochs_interval=max(1, int(NUM_EPOCHS / 1))
    )


    print("")
    orch.print_results()
    print("")

    mlp_persister = MLPPersister()
    mlp_persister.save(mlp, "models/mlp_caracteres.json")
