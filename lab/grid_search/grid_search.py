# ===== IMPORTS ==============================

from datetime import datetime
from itertools import product

from src.dataset_loader import DatasetLoader
from src.funcs import sigmoide_bipolar, deriv_sigmoide_bipolar
from src.models import Layer, MLP
from src.orchestrator import Orchestrator
from src.mlp_persister import MLPPersister
from lab.grid_search.reporter import GridSearchReporter
from lab.grid_search.initial_weights import get_grid_search_initial_weights

# =====


def pipeline(
    train_x: list[list[float]],
    train_y: list[list[float]],
    test_x: list[list[float]],
    test_y: list[list[float]],
    epochs: int,
    target_mse: float | None,
    hidden_neurons: int,
    learn_rate: float,
    momentum: float,
    patience: int | None,
    output_dir: str,
    min_acc_threshold: float | None = None,
) -> tuple[Orchestrator, str | None]:
    """
    Constrói a arquitetura, injeta pesos determinísticos e executa o ciclo de treino.
    Persiste o modelo somente se atingir o limiar mínimo de acurácia configurado.
    """
    num_inputs = len(train_x[0])
    num_outputs = len(train_y[0])

    hidden_layer = Layer(
        num_inputs=num_inputs,
        num_neurons=hidden_neurons,
        activation_func=sigmoide_bipolar,
        derivative_activation=deriv_sigmoide_bipolar,
    )

    init_weights = get_grid_search_initial_weights(
        hidden_neurons=hidden_neurons,
        num_inputs=num_inputs,
        num_outputs=num_outputs,
    )

    mlp = MLP(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        learn_rate=learn_rate,
        hidden_layers=[hidden_layer],
        output_activation=sigmoide_bipolar,
        output_deriv_activation=deriv_sigmoide_bipolar,
        momentum=momentum,
        initial_weights=init_weights,
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

    # Acurácia de validação do melhor estado registrado
    best_val_acc_pct = orch.best_val_acc * 100

    saved_file: str | None = None
    if min_acc_threshold is not None and best_val_acc_pct < min_acc_threshold:
        print(
            f"Checkpoint descartado: acurácia ({best_val_acc_pct:.2f}%) "
            f"inferior ao corte de {min_acc_threshold:.1f}%."
        )
    else:
        saved_file = MLPPersister.save_checkpoint(orch, dir_path=output_dir, prefix="gs")

    return orch, saved_file


# =====


if __name__ == "__main__":

    # == Diretório da Execução ====
    INIT_TIME = datetime.now().strftime("%Y_%m_%d_%H_%M")
    OUTPUT_DIR = f"models/grid_search/{INIT_TIME}/"

    # == Hiperparâmetros Fixos e Critérios ====
    MAX_EPOCHS = 150
    TARGET_MSE = 0.001
    MIN_ACC_THRESHOLD = 80.0

    # == Grade de Busca ====
    HIDDEN_NEURONS = [80]
    VAL_RATIOS = [0.30]
    LEARN_RATE = [0.003]
    MOMENTUM = [0.80]
    PATIENCE = [4]

    reporter = GridSearchReporter(output_dir=OUTPUT_DIR)

    # Cache do dataset particionado por proporção de teste (evita releitura de disco)
    dataset_cache: dict[float, tuple] = {}

    combinations = list(
        product(
            VAL_RATIOS,
            HIDDEN_NEURONS,
            LEARN_RATE,
            MOMENTUM,
            PATIENCE,
        )
    )
    total_combinations = len(combinations)

    for current, (val_ratio, hidden, lr, momentum, patience) in enumerate(combinations, start=1):
        if val_ratio not in dataset_cache:
            dataset_cache[val_ratio] = DatasetLoader.load_caracteres_completo(val_ratio=val_ratio)

        train_x, train_y, test_x, test_y, _ = dataset_cache[val_ratio]

        print(
            f"\n[RODADA {current}/{total_combinations}] "
            f"Ratio={val_ratio} | "
            f"H={hidden} | "
            f"LR={lr} | "
            f"M={momentum} | "
            f"P={patience}"
        )

        orch, saved_file = pipeline(
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
            epochs=MAX_EPOCHS,
            target_mse=TARGET_MSE,
            hidden_neurons=hidden,
            learn_rate=lr,
            momentum=momentum,
            patience=patience,
            output_dir=OUTPUT_DIR,
            min_acc_threshold=MIN_ACC_THRESHOLD,
        )

        reporter.record_trial(
            round_num=current,
            hidden_neurons=hidden,
            learn_rate=lr,
            momentum=momentum,
            patience=patience,
            orch=orch,
            val_ratio=val_ratio,
            checkpoint_path=saved_file,
        )

    # == Exportar Resumo e Exibir Ranking ====
    reporter.save_csv()
    reporter.print_ranking(top_k=10)
    reporter.finalize_directory_name()
