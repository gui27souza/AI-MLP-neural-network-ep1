# ===== IMPORTS ==============================

import os

from src.dataset_loader import DatasetLoader
from src.funcs import sigmoide_bipolar, deriv_sigmoide_bipolar
from src.models import Layer, MLP
from src.orchestrator import Orchestrator
from src.mlp_persister import MLPPersister

# =====

# ===== ENV PARAMS ==============================

NUM_EPOCHS = int(os.environ["NUM_EPOCHS"])
LEARN_RATE = float(os.environ["LEARN_RATE"])
HIDDEN_NEURONS = int(os.environ["HIDDEN_NEURONS"])
MOMENTUM = float(os.environ["MOMENTUM"])

_target_mse_env = os.getenv("TARGET_MSE")
TARGET_MSE = float(_target_mse_env) if _target_mse_env else None

_patience_env = os.getenv("PATIENCE")
PATIENCE = int(_patience_env) if _patience_env else None

PRINT_INTERVAL = int(os.environ["PRINT_INTERVAL"])

# =====


def build_mlp(num_inputs: int, num_outputs: int) -> MLP:
    """Instancia a arquitetura da rede neural para classificação dos caracteres."""

    hidden_layer = Layer(
        num_inputs=num_inputs,
        num_neurons=HIDDEN_NEURONS,
        activation_func=sigmoide_bipolar,
        derivative_activation=deriv_sigmoide_bipolar,
    )

    return MLP(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        learn_rate=LEARN_RATE,
        hidden_layers=[hidden_layer],
        output_activation=sigmoide_bipolar,
        output_deriv_activation=deriv_sigmoide_bipolar,
        momentum=MOMENTUM,
    )


if __name__ == "__main__":

    # == Carregamento dos dados ====
    train_x, train_y, test_x, test_y, _ = DatasetLoader.load_caracteres_completo()

    # == Construção do modelo ====
    mlp = build_mlp(num_inputs=len(train_x[0]), num_outputs=len(train_y[0]))

    # == Preparação do orquestrador ====
    orch = Orchestrator(
        mlp=mlp,
        epochs=NUM_EPOCHS,
        inputs=train_x,
        targets=train_y,
        val_inputs=test_x,
        val_targets=test_y,
        target_mse=TARGET_MSE,
        patience=PATIENCE,
    )

    # == Execução do treinamento com salvamento resiliente ====
    try:
        orch.run(print_epochs_interval=PRINT_INTERVAL)
        print("")
        orch.print_results()
        print("")
    except KeyboardInterrupt:
        print("\n\nTreinamento interrompido manualmente (Ctrl+C). Salvando progresso...")
    finally:
        MLPPersister.save_checkpoint(orch)
