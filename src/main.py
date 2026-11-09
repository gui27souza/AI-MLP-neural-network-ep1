import os
from datetime import datetime

from src.dataset_loader import DatasetLoader
from src.funcs import sigmoide_bipolar, deriv_sigmoide_bipolar
from src.models import Layer, MLP
from src.orchestrator import Orchestrator
from src.mlp_persister import MLPPersister


NUM_EPOCHS = int(os.environ["NUM_EPOCHS"])
LEARN_RATE = float(os.environ["LEARN_RATE"])
HIDDEN_NEURONS = int(os.environ["HIDDEN_NEURONS"])


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
    )


def save_model_checkpoint(mlp: MLP, orch: Orchestrator) -> None:
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

        metrics = {
            "epochs": orch.total_epochs,
            "duration_seconds": round(duration, 2),
            "train_mse": orch.history["train_mse"][-1] if orch.history["train_mse"] else None,
            "train_acc": round(train_acc_pct, 2),
            "val_mse": round(val_mse, 5),
            "val_acc": round(val_acc_pct, 2),
        }

        hidden_desc = "h" + "_".join(str(layer.num_neurons) for layer in mlp.hidden_layers)
        filename = f"mlp_caracteres_{hidden_desc}_e{orch.total_epochs}_acc{val_acc_pct:.1f}_mse{val_mse:.3f}.json"
        filepath = f"models/{filename}"

        MLPPersister.save(mlp, filepath, metrics=metrics)

    except Exception as e:
        print(f"não foi possível salvar o modelo treinado: {e}")


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
