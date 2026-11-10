# ===== IMPORTS ==============================

import os
from datetime import datetime

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


def save_model_checkpoint(mlp: MLP, orch: Orchestrator, path="models/") -> None:
    """Extrai métricas do orquestrador e salva o modelo persistido com nome semântico."""

    try:

        duration = getattr(orch, "delta_ts", None)
        if duration is None and hasattr(orch, "start_ts"):
            duration = datetime.now().timestamp() - orch.start_ts
        else:
            duration = duration or 0.0

        val_acc_pct = (orch.history["val_acc"][-1] * 100) if orch.history["val_acc"] else 0.0
        train_acc_pct = (orch.history["train_acc"][-1] * 100) if orch.history["train_acc"] else 0.0
        val_mse = orch.history["val_mse"][-1] if orch.history["val_mse"] else 0.0

        # Identifica os picos históricos de validação (melhor época antes de eventual overfitting)
        best_val_acc = 0.0
        best_val_acc_epoch = 0
        if orch.history["val_acc"]:
            best_val_acc = max(orch.history["val_acc"])
            best_val_acc_epoch = orch.history["val_acc"].index(best_val_acc) + 1

        best_val_mse = 0.0
        best_val_mse_epoch = 0
        if orch.history["val_mse"]:
            best_val_mse = min(orch.history["val_mse"])
            best_val_mse_epoch = orch.history["val_mse"].index(best_val_mse) + 1

        # Se os melhores pesos foram restaurados pelo Early Stopping / Validação
        weights_restored = orch.best_weights is not None
        saved_epoch = orch.best_epoch if weights_restored else orch.total_epochs
        saved_val_mse = orch.best_val_mse if weights_restored else val_mse
        saved_val_acc_pct = (orch.best_val_acc * 100) if weights_restored else val_acc_pct

        metrics = {
            "dataset_info": {
                "train_samples": len(orch.inputs),
                "val_samples": len(orch.val_inputs) if orch.val_inputs else 0,
            },
            "training_params": {
                "max_epochs": orch.epochs,
                "target_mse": orch.target_mse,
                "patience": orch.patience,
            },
            "stopping_info": {
                "stopped_early": orch.stopped_early,
                "stop_reason": orch.stop_reason or "Executou todas as épocas programadas.",
                "weights_restored_from_best": weights_restored,
                "saved_epoch": saved_epoch,
            },
            "epochs_executed": orch.total_epochs,
            "duration_seconds": round(duration, 2),
            "train_mse": orch.history["train_mse"][-1] if orch.history["train_mse"] else None,
            "train_acc": round(train_acc_pct, 2),
            "val_mse": round(val_mse, 5),
            "val_acc": round(val_acc_pct, 2),
            "best_val_acc": round(best_val_acc * 100, 2),
            "best_val_acc_epoch": best_val_acc_epoch,
            "best_val_mse": round(best_val_mse, 5),
            "best_val_mse_epoch": best_val_mse_epoch,
        }

        hidden_desc = "h" + "_".join(str(layer.num_neurons) for layer in mlp.hidden_layers)
        filename = (
            f"mlp_caracteres_{hidden_desc}_e{saved_epoch}"
            f"_acc{saved_val_acc_pct:.1f}_mse{saved_val_mse:.3f}.json"
        )
        filepath = f"{path}{filename}"

        MLPPersister.save(mlp, filepath, metrics=metrics)

    except Exception as e:
        print(f"não foi possível salvar o modelo treinado: {e}")

# =====

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
        orch.run(print_epochs_interval=max(1, int(NUM_EPOCHS / 10)))
        print("")
        orch.print_results()
        print("")
    except KeyboardInterrupt:
        print("\n\nTreinamento interrompido manualmente (Ctrl+C). Salvando progresso...")
    finally:
        save_model_checkpoint(mlp, orch)
