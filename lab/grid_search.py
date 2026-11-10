# ===== IMPORTS ==============================

from datetime import datetime

from src.dataset_loader import DatasetLoader
from src.funcs import sigmoide_bipolar, deriv_sigmoide_bipolar
from src.models import Layer, MLP
from src.orchestrator import Orchestrator

from src.main import save_model_checkpoint

# =====

INIT_TIME = datetime.now().strftime("%Y_%m_%d_%H_%M")
PATH = f"models/grid_search/{INIT_TIME}/"

def pipeline(
    train_x, train_y, test_x, test_y,
    epochs, target_mse,
    hidden_neurons, learn_rate, momentum, patience
):

    hidden_layer = Layer(
        num_inputs=len(train_x[0]),
        num_neurons=hidden_neurons,
        activation_func=sigmoide_bipolar,
        derivative_activation=deriv_sigmoide_bipolar,
    )

    mlp = MLP(
        num_inputs=len(train_x[0]),
        num_outputs=len(train_y[0]),
        learn_rate=learn_rate,
        hidden_layers=[hidden_layer],
        output_activation=sigmoide_bipolar,
        output_deriv_activation=deriv_sigmoide_bipolar,
        momentum=momentum,
    )

    orch = Orchestrator(
        mlp=mlp,
        epochs=epochs,
        inputs=train_x,
        targets=train_y,
        val_inputs=test_x,
        val_targets=test_y,
        target_mse=target_mse,
        patience=patience,
    )

    orch.run(print_epochs_interval=2)
    print("")
    orch.print_results()
    print("")
    save_model_checkpoint(
        mlp, orch,
        path=PATH,
    )


if __name__ == "__main__":

    MAX_EPOCHS = 150
    TARGET_MSE = 0.001

    HIDDEN_NEURONS = [16, 32, 64]
    LEARN_RATE = [0.005, 0.01, 0.025]
    MOMENTUM = [0.4, 0.8]
    PATIENCE = [3, 7]

    # == Carregamento dos dados ====
    train_x, train_y, test_x, test_y, _ = DatasetLoader.load_caracteres_completo()

    total_combinations = (
        len(HIDDEN_NEURONS) *
        len(LEARN_RATE) *
        len(MOMENTUM) *
        len(PATIENCE)
    )
    current = 1

    for hidden_neurons in HIDDEN_NEURONS:
        for learn_rate in LEARN_RATE:
            for momentum in MOMENTUM:
                for patience in PATIENCE:

                    print(
                        f"\n[RODADA {current}/{total_combinations}] "
                        f"H={hidden_neurons} | "
                        f"LR={learn_rate} | "
                        f"M={momentum} | "
                        f"P={patience}"
                    )

                    pipeline(
                        train_x, train_y, test_x, test_y,
                        epochs=MAX_EPOCHS,
                        target_mse=TARGET_MSE,
                        hidden_neurons=hidden_neurons,
                        learn_rate=learn_rate,
                        momentum=momentum,
                        patience=patience
                    )

                    current += 1
