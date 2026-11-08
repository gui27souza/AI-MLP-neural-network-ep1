import os
from datetime import datetime

from src.dataset_loader import DatasetLoader

from src.funcs import (
    sigmoide_bipolar, deriv_sigmoide_bipolar,
)
from src.models import (
    Layer, MLP,
)
from src.orchestrator import Orchestrator
from src.mlp_persister import MLPPersister


NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 20))


if __name__ == "__main__":

    train_inputs, train_targets, test_inputs, test_targets, class_labels = (
        DatasetLoader.load_caracteres_completo()
    )

    layer = Layer(
        num_inputs=len(train_inputs[0]),
        num_neurons=32,
        activation_func=sigmoide_bipolar,
        derivative_activation=deriv_sigmoide_bipolar,
    )

    mlp = MLP(
        num_inputs=len(train_inputs[0]),
        num_outputs=len(train_targets[0]),
        learn_rate=0.015,
        hidden_layers=[layer],
        output_activation=sigmoide_bipolar,
        output_deriv_activation=deriv_sigmoide_bipolar,
    )

    orch = Orchestrator(
        mlp=mlp,
        epochs=NUM_EPOCHS,
        inputs=train_inputs,
        targets=train_targets,
        val_inputs=test_inputs,
        val_targets=test_targets,
    )

    orch.run(
        print_epochs_interval=max(1, int(NUM_EPOCHS / 10))
    )

    print("")
    orch.print_results()
    print("")

    # Prepara resumo de métricas para persistir no modelo
    val_acc_pct = (orch.history["val_acc"][-1] * 100) if orch.history["val_acc"] else 0.0
    train_acc_pct = (orch.history["train_acc"][-1] * 100) if orch.history["train_acc"] else 0.0
    val_mse = orch.history["val_mse"][-1] if orch.history["val_mse"] else 0.0

    metrics = {
        "epochs": orch.total_epochs,
        "duration_seconds": round(orch.delta_ts, 2),
        "train_mse": orch.history["train_mse"][-1] if orch.history["train_mse"] else None,
        "train_acc": round(train_acc_pct, 2),
        "val_mse": round(val_mse, 5),
        "val_acc": round(val_acc_pct, 2),
    }

    # Gera nome limpo com arquitetura (ex: h32), épocas, acurácia e MSE (anti-colisão com _v2 automático)
    hidden_desc = "h" + "_".join(str(layer.num_neurons) for layer in mlp.hidden_layers)
    filename = f"mlp_caracteres_{hidden_desc}_e{NUM_EPOCHS}_acc{val_acc_pct:.1f}_mse{val_mse:.3f}.json"
    filepath = f"models/{filename}"

    MLPPersister.save(mlp, filepath, metrics=metrics)
