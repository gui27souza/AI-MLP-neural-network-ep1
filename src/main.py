# ===== IMPORTS ==============================

import os

from src.data_augmentation import augment_bipolar_dataset
from src.dataset_loader import DatasetLoader
from src.funcs import sigmoide_bipolar, deriv_sigmoide_bipolar
from src.initial_weights import get_initial_weights
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
VAL_RATIO = float(os.environ["VAL_RATIO"])

_seed_env = os.getenv("SEED", "42")
SEED = int(_seed_env) if _seed_env.lower() not in ("none", "") else None

AUGMENT_DATA = os.getenv("AUGMENT_DATA", "0").lower() in ("1", "true", "yes")
NOISE_RATE = float(os.getenv("NOISE_RATE", "0.05"))
AUGMENT_COPIES = int(os.getenv("AUGMENT_COPIES", "1"))

# =====


if __name__ == "__main__":

    # == Carregamento dos dados ====
    train_x, train_y, test_x, test_y, _ = DatasetLoader.load_caracteres_completo(val_ratio=VAL_RATIO)

    if AUGMENT_DATA:
        train_x, train_y = augment_bipolar_dataset(
            train_x,
            train_y,
            noise_rate=NOISE_RATE,
            copies_per_sample=AUGMENT_COPIES,
            seed=SEED if SEED is not None else 42,
        )

    num_inputs = len(train_x[0])
    num_outputs = len(train_y[0])

    # == Construção do modelo ====
    hidden_layer = Layer(
        num_inputs=num_inputs,
        num_neurons=HIDDEN_NEURONS,
        activation_func=sigmoide_bipolar,
        derivative_activation=deriv_sigmoide_bipolar,
    )

    init_weights = (
        get_initial_weights(
            hidden_neurons=HIDDEN_NEURONS,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            seed=SEED if SEED is not None else 42,
        )
        if SEED is not None
        else None
    )

    mlp = MLP(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        learn_rate=LEARN_RATE,
        hidden_layers=[hidden_layer],
        output_activation=sigmoide_bipolar,
        output_deriv_activation=deriv_sigmoide_bipolar,
        momentum=MOMENTUM,
        initial_weights=init_weights,
    )

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
        seed=SEED,
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
        # Salva o checkpoint final e registra o caminho gerado para automações
        # Necessário para que o make full_pipeline funcione devidamente de forma automatizada
        saved_path = MLPPersister.save_checkpoint(orch)
        if saved_path:
            with open("models/.latest_model.txt", "w", encoding="utf-8") as f:
                f.write(saved_path.strip())
