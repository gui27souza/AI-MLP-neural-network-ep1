# ===== IMPORTS ==============================

from datetime import datetime

from src.dataset_loader import DatasetLoader
from src.funcs import sigmoide_bipolar, deriv_sigmoide_bipolar
from src.models import Layer, MLP
from src.orchestrator import Orchestrator
from src.mlp_persister import MLPPersister
from lab.grid_search.reporter import GridSearchReporter

# =====

INIT_TIME = datetime.now().strftime("%Y_%m_%d_%H_%M")
PATH = f"models/grid_search/{INIT_TIME}/"


def pipeline(
    train_x, train_y, test_x, test_y,
    epochs, target_mse,
    hidden_neurons, learn_rate, momentum, patience,
    min_acc_threshold: float | None = None,
) -> tuple[Orchestrator, str | None]:

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

    orch.run(print_epochs_interval=5)
    print("")
    orch.print_results()
    print("")

    best_val_acc_pct = (orch.best_val_acc * 100) if orch.best_weights is not None else (
        (orch.history["val_acc"][-1] * 100) if orch.history["val_acc"] else 0.0
    )

    if min_acc_threshold is not None and best_val_acc_pct < min_acc_threshold:
        print(
            f"Checkpoint descartado: acuracia ({best_val_acc_pct:.2f}%) "
            f"inferior ao corte de {min_acc_threshold:.1f}%."
        )
        saved_file = None
    else:
        saved_file = MLPPersister.save_checkpoint(orch, dir_path=PATH, prefix="gs")

    return orch, saved_file


if __name__ == "__main__":

    MAX_EPOCHS = 150
    TARGET_MSE = 0.001
    MIN_ACC_THRESHOLD = 80.0

    HIDDEN_NEURONS = [80]
    VAL_RATIOS = [0.30]
    LEARN_RATE = [0.003]
    MOMENTUM = [0.80]
    PATIENCE = [4]

    reporter = GridSearchReporter(output_dir=PATH)

    total_combinations = (
        len(VAL_RATIOS) *
        len(HIDDEN_NEURONS) *
        len(LEARN_RATE) *
        len(MOMENTUM) *
        len(PATIENCE)
    )
    current = 1

    for val_ratio in VAL_RATIOS:

        # Carrega o dataset particionado conforme a proporção de validação da rodada
        train_x, train_y, test_x, test_y, _ = DatasetLoader.load_caracteres_completo(val_ratio=val_ratio)

        for hidden_neurons in HIDDEN_NEURONS:
            for learn_rate in LEARN_RATE:
                for momentum in MOMENTUM:
                    for patience in PATIENCE:

                        print(
                            f"\n[RODADA {current}/{total_combinations}] "
                            f"Ratio={val_ratio} | "
                            f"H={hidden_neurons} | "
                            f"LR={learn_rate} | "
                            f"M={momentum} | "
                            f"P={patience}"
                        )

                        orch, saved_file = pipeline(
                            train_x, train_y, test_x, test_y,
                            epochs=MAX_EPOCHS,
                            target_mse=TARGET_MSE,
                            hidden_neurons=hidden_neurons,
                            learn_rate=learn_rate,
                            momentum=momentum,
                            patience=patience,
                            min_acc_threshold=MIN_ACC_THRESHOLD,
                        )

                        reporter.record_trial(
                            round_num=current,
                            hidden_neurons=hidden_neurons,
                            learn_rate=learn_rate,
                            momentum=momentum,
                            patience=patience,
                            orch=orch,
                            val_ratio=val_ratio,
                            checkpoint_path=saved_file,
                        )

                        current += 1

    # == Exportar Resumo e Exibir Ranking ====
    reporter.save_csv()
    reporter.print_ranking(top_k=10)
    reporter.finalize_directory_name()
