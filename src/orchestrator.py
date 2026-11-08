from datetime import datetime, timedelta

from .models import MLP


class Orchestrator:

    start_ts: float
    end_ts: float
    delta_ts: float
    duration: timedelta

    def __init__(
        self,
        mlp: MLP,
        epochs: int,
        inputs: list[list[float]],
        targets: list[list[float]],
        val_inputs: list[list[float]] | None = None,
        val_targets: list[list[float]] | None = None,
    ):
        self.mlp = mlp
        self.epochs = epochs
        self.inputs = inputs
        self.targets = targets
        self.val_inputs = val_inputs
        self.val_targets = val_targets

        self.total_epochs = 0

        # Histórico de métricas coletadas ao fim de cada época
        self.history: dict[str, list[float]] = {
            "train_mse": [],
            "train_acc": [],
            "val_mse": [],
            "val_acc": [],
        }

    def run(
        self,
        print_epochs_interval: int = 1,
        verbose_samples: bool = False,
    ):
        self.start_ts = datetime.now().timestamp()
        has_val = self.val_inputs is not None and self.val_targets is not None

        # Cabeçalho da tabela de progresso
        print(
            f"{'ÉPOCA':<8} | {'MSE TREINO':<12} | {'ACC TREINO':<12} | {'MSE VAL':<12} | {'ACC VAL'}"
        )
        print("-" * 68)

        for epoch in range(1, self.epochs + 1):

            # Executa a época padrão a padrão (ajuste de pesos via Backpropagation)
            self.mlp.run_epoch(
                inputs_list=self.inputs,
                targets=self.targets,
                print_info=verbose_samples,
            )

            # Avalia métricas no conjunto de treino com o predict puro
            train_mse, train_acc = self.mlp.evaluate(self.inputs, self.targets)
            self.history["train_mse"].append(train_mse)
            self.history["train_acc"].append(train_acc)

            # Avalia métricas no conjunto de validação/teste (se fornecido)
            val_mse_str = "N/A"
            val_acc_str = "N/A"
            if has_val and self.val_inputs and self.val_targets:
                val_mse, val_acc = self.mlp.evaluate(self.val_inputs, self.val_targets)
                self.history["val_mse"].append(val_mse)
                self.history["val_acc"].append(val_acc)
                val_mse_str = f"{val_mse:.5f}"
                val_acc_str = f"{val_acc * 100:.2f}%"

            self.total_epochs = epoch

            # Imprime linha resumida no terminal conforme o intervalo desejado
            should_print = (
                epoch == 1
                or epoch % print_epochs_interval == 0
                or epoch == self.epochs
            )
            if should_print:
                print(
                    f"{epoch:<8} | {train_mse:<12.5f} | {train_acc * 100:<10.2f}% | "
                    f"{val_mse_str:<12} | {val_acc_str}"
                )

        self.end_ts = datetime.now().timestamp()
        self.delta_ts = self.end_ts - self.start_ts
        self.duration = timedelta(seconds=self.delta_ts)

    def print_results(self):
        print("=" * 68)
        print(f"Tempo total de execução: {self.delta_ts:.2f}s ({self.duration})")
        print(f"Total de épocas executadas: {self.total_epochs}")
        if self.history["train_mse"]:
            print(
                f"MSE Treino Final: {self.history['train_mse'][-1]:.6f} | "
                f"Acc Treino Final: {self.history['train_acc'][-1] * 100:.2f}%"
            )
        if self.history["val_mse"]:
            print(
                f"MSE Val Final:    {self.history['val_mse'][-1]:.6f} | "
                f"Acc Val Final:    {self.history['val_acc'][-1] * 100:.2f}%"
            )
        print("=" * 68)
