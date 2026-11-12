from datetime import datetime, timedelta
import random

from src.core.models import MLP


class Orchestrator:
    """Gerencia o ciclo de treinamento da MLP (epocas, shuffling, early stopping e metricas)."""

    def __init__(
        self,
        mlp: MLP,
        epochs: int,
        inputs: list[list[float]],
        targets: list[list[float]],
        val_inputs: list[list[float]] | None = None,
        val_targets: list[list[float]] | None = None,
        target_mse: float | None = None,
        patience: int | None = None,
        seed: int | None = 42,
    ):
        self.mlp = mlp
        self.epochs = epochs
        self.inputs = inputs
        self.targets = targets
        self.val_inputs = val_inputs
        self.val_targets = val_targets
        self.target_mse = target_mse
        self.patience = patience
        self.seed = seed
        self.rng = random.Random(seed) if seed is not None else random.Random()

        self.start_ts: float = 0.0
        self.end_ts: float = 0.0
        self.delta_ts: float = 0.0
        self.duration: timedelta = timedelta(0)

        self.total_epochs: int = 0
        self.best_epoch: int = 0
        self.best_val_mse: float = float("inf")
        self.best_val_acc: float = 0.0
        self.best_weights: dict | None = None
        self.stopped_early: bool = False
        self.stop_reason: str = ""
        self._patience_counter: int = 0

        self.initial_weights = getattr(self.mlp, "initial_weights", None) or self.mlp.get_weights()
        self.history: dict[str, list[float]] = {
            "train_mse": [],
            "train_acc": [],
            "val_mse": [],
            "val_acc": [],
        }

    # =====================================
    # Métodos Auxiliares do Ciclo de Época
    # =====================================

    def _train_epoch(self, index: list[int], verbose: bool):
        self.rng.shuffle(index)
        inputs_shuffled = [self.inputs[i] for i in index]
        targets_shuffled = [self.targets[i] for i in index]
        self.mlp.run_epoch(inputs_shuffled, targets_shuffled, print_info=verbose)

    def _evaluate_train(self) -> tuple[float, float]:
        mse, acc = self.mlp.evaluate(self.inputs, self.targets)
        self.history["train_mse"].append(mse)
        self.history["train_acc"].append(acc)
        return mse, acc

    def _evaluate_val(self, epoch: int) -> tuple[float | None, float | None]:
        if not self.val_inputs or not self.val_targets:
            return None, None

        mse, acc = self.mlp.evaluate(self.val_inputs, self.val_targets)
        self.history["val_mse"].append(mse)
        self.history["val_acc"].append(acc)

        is_better = (acc > self.best_val_acc) or (acc == self.best_val_acc and mse < self.best_val_mse)
        if is_better:
            self.best_val_acc = acc
            self.best_val_mse = mse
            self.best_epoch = epoch
            self.best_weights = self.mlp.get_weights()
            self._patience_counter = 0
        else:
            self._patience_counter += 1

        return mse, acc

    def _check_early_stop(self, train_mse: float) -> str | None:
        if self.target_mse is not None and train_mse <= self.target_mse:
            return f"MSE de treino ({train_mse:.6f}) atingiu o erro alvo ({self.target_mse:.6f})."

        has_val = bool(self.val_inputs and self.val_targets)
        if has_val and self.patience is not None and self._patience_counter >= self.patience:
            return f"Early Stopping acionado apos {self.patience} epocas sem melhora na validacao."

        return None

    def _print_epoch(self, epoch: int, train_mse: float, train_acc: float, val_mse: float | None, val_acc: float | None):
        val_mse_str = f"{val_mse:.5f}" if val_mse is not None else "N/A"
        val_acc_str = f"{val_acc * 100:.2f}%" if val_acc is not None else "N/A"
        print(f"{epoch:<8} | {train_mse:<12.5f} | {train_acc * 100:<10.2f}% | {val_mse_str:<12} | {val_acc_str}")

    # =============================
    # Execução Principal e Sumário
    # =============================

    def run(self, print_epochs_interval: int = 1, verbose_samples: bool = False):
        """Executa o loop de epocas, avaliacao e checagem de criterios de parada."""
        self.start_ts = datetime.now().timestamp()
        self._patience_counter = 0

        print(f"{'ÉPOCA':<8} | {'MSE TREINO':<12} | {'ACC TREINO':<12} | {'MSE VAL':<12} | {'ACC VAL'}")
        print("-" * 68)

        index = list(range(len(self.inputs)))

        for epoch in range(1, self.epochs + 1):
            self.total_epochs = epoch
            self._train_epoch(index, verbose=verbose_samples)

            train_mse, train_acc = self._evaluate_train()
            val_mse, val_acc = self._evaluate_val(epoch)

            should_print = (epoch == 1 or epoch % print_epochs_interval == 0 or epoch == self.epochs)
            if should_print:
                self._print_epoch(epoch, train_mse, train_acc, val_mse, val_acc)

            stop_reason = self._check_early_stop(train_mse)
            if stop_reason:
                self.stopped_early = True
                self.stop_reason = stop_reason
                if not should_print:
                    self._print_epoch(epoch, train_mse, train_acc, val_mse, val_acc)
                break

        if self.best_weights is not None:
            self.mlp.set_weights(self.best_weights)

        self.end_ts = datetime.now().timestamp()
        self.delta_ts = self.end_ts - self.start_ts
        self.duration = timedelta(seconds=self.delta_ts)

    def print_results(self):
        """Exibe o sumario final do treinamento."""
        print("=" * 68)
        print(f"Tempo total de execução: {self.delta_ts:.2f}s ({self.duration})")
        print(f"Total de épocas executadas: {self.total_epochs}")
        if self.stopped_early:
            print(f"Critério de Parada: {self.stop_reason}")
        if self.best_weights is not None:
            print(f"Melhor Época: {self.best_epoch} | MSE Val: {self.best_val_mse:.6f} | Acc Val: {self.best_val_acc * 100:.2f}% (pesos restaurados na MLP)")
        if self.history["train_mse"]:
            print(f"MSE Treino Final: {self.history['train_mse'][-1]:.6f} | Acc Treino Final: {self.history['train_acc'][-1] * 100:.2f}%")
        if self.history["val_mse"]:
            print(f"MSE Val Final:    {self.history['val_mse'][-1]:.6f} | Acc Val Final:    {self.history['val_acc'][-1] * 100:.2f}%")
        print("=" * 68)
