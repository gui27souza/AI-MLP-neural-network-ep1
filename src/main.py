from .models import Layer, MLP
from .funcs import sigmoide_binaria, deriv_sigmoide_binaria

NUM_EPOCHS = 10

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

    for i in range(1, NUM_EPOCHS+1):
        print(f"\n=== Epoch {i}")
        mlp.run_epoch(inputs, targets_AND)
        print("==========")
