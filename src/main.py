# ===== IMPORTS ==============================

from src.core.funcs import sigmoide_bipolar, deriv_sigmoide_bipolar
from src.core.initial_weights import get_initial_weights
from src.core.models import Layer, MLP
from src.data.dataset_loader import DatasetLoader
from src.persistence.mlp_persister import MLPPersister
from src.training.config import TrainConfig
from src.training.orchestrator import Orchestrator

# =====


if __name__ == "__main__":
    cfg = TrainConfig.from_env()

    # == Carregamento e aumento dos dados ====
    train_x, train_y, test_x, test_y, _ = DatasetLoader.load_caracteres_completo(val_ratio=cfg.val_ratio)
    train_x, train_y = cfg.augment.apply(train_x, train_y, seed=cfg.seed if cfg.seed is not None else 42)

    num_inputs = len(train_x[0])
    num_outputs = len(train_y[0])

    # == Construção do modelo ====
    hidden_layer = Layer(
        num_inputs=num_inputs,
        num_neurons=cfg.hidden_neurons,
        activation_func=sigmoide_bipolar,
        derivative_activation=deriv_sigmoide_bipolar,
    )

    init_weights = (
        get_initial_weights(
            hidden_neurons=cfg.hidden_neurons,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            seed=cfg.seed if cfg.seed is not None else 42,
        )
        if cfg.seed is not None
        else None
    )

    mlp = MLP(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        learn_rate=cfg.learn_rate,
        hidden_layers=[hidden_layer],
        output_activation=sigmoide_bipolar,
        output_deriv_activation=deriv_sigmoide_bipolar,
        momentum=cfg.momentum,
        initial_weights=init_weights,
    )

    # == Preparação do orquestrador ====
    orch = Orchestrator(
        mlp=mlp,
        epochs=cfg.epochs,
        inputs=train_x,
        targets=train_y,
        val_inputs=test_x,
        val_targets=test_y,
        target_mse=cfg.target_mse,
        patience=cfg.patience,
        seed=cfg.seed,
    )

    # == Execução do treinamento com salvamento resiliente ====
    try:
        orch.run(print_epochs_interval=cfg.print_interval)
        print("")
        orch.print_results()
        print("")
    except KeyboardInterrupt:
        print("\n\nTreinamento interrompido manualmente (Ctrl+C). Salvando progresso...")
    finally:
        saved_path = MLPPersister.save_checkpoint(orch)
        if saved_path:
            with open("models/.latest_model.txt", "w", encoding="utf-8") as f:
                f.write(saved_path.strip())
