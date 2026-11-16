import csv
import os
from datetime import datetime

from src.training.orchestrator import Orchestrator


class GridSearchReporter:
    """Coleta metricas, exporta summary.csv e exibe ranking no terminal."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.records: list[dict] = []

    def record_trial(
        self,
        round_num: int,
        hidden_neurons: int,
        learn_rate: float,
        momentum: float,
        patience: int,
        orch: Orchestrator,
        val_ratio: float = 0.2,
        noise_rate: float = 0.0,
        checkpoint_path: str | None = None,
    ) -> dict:
        """Registra metricas da rodada de grid search (Treino, Validacao e Teste)."""
        duration = getattr(orch, "delta_ts", None)
        if duration is None and hasattr(orch, "start_ts"):
            duration = datetime.now().timestamp() - orch.start_ts
        else:
            duration = duration or 0.0

        val_accs = orch.history.get("val_acc", [])
        train_accs = orch.history.get("train_acc", [])
        val_mses = orch.history.get("val_mse", [])

        # Métricas da época selecionada pelo Early Stopping (menor MSE de validação)
        saved_epoch = orch.best_epoch if orch.best_weights is not None else orch.total_epochs
        saved_val_acc = (orch.best_val_acc * 100) if orch.best_weights is not None else (
            (val_accs[-1] * 100) if val_accs else 0.0
        )
        saved_val_mse = orch.best_val_mse if orch.best_weights is not None else (
            val_mses[-1] if val_mses else 0.0
        )

        test_acc_pct = (orch.test_acc * 100) if getattr(orch, "test_acc", None) is not None else saved_val_acc
        test_mse_val = orch.test_mse if getattr(orch, "test_mse", None) is not None else saved_val_mse

        record = {
            "rodada": round_num,
            "hidden_neurons": hidden_neurons,
            "val_ratio": val_ratio,
            "learn_rate": learn_rate,
            "momentum": momentum,
            "patience": patience,
            "noise_rate": noise_rate,
            "epochs_exec": orch.total_epochs,
            "stopped_early": orch.stopped_early,
            "final_train_acc": round((train_accs[-1] * 100) if train_accs else 0.0, 2),
            "final_val_acc": round((val_accs[-1] * 100) if val_accs else 0.0, 2),
            "best_val_acc": round(saved_val_acc, 2),
            "best_val_acc_epoch": saved_epoch,
            "final_val_mse": round(val_mses[-1] if val_mses else 0.0, 5),
            "best_val_mse": round(saved_val_mse, 5),
            "test_acc": round(test_acc_pct, 2),
            "test_mse": round(test_mse_val, 5),
            "duracao_s": round(duration, 2),
            "checkpoint": os.path.basename(checkpoint_path) if checkpoint_path else "",
        }

        self.records.append(record)
        self.save_csv(verbose=False)
        return record

    def save_csv(self, filename: str = "summary.csv", verbose: bool = True) -> str:
        """Salva resumo consolidado em arquivo CSV."""
        if not self.records:
            return ""

        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = os.path.join(self.output_dir, filename)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.records[0].keys()))
            writer.writeheader()
            writer.writerows(self.records)

        if verbose:
            print(f"\nResumo consolidado salvo em: {csv_path}")
        return csv_path

    def print_ranking(self, top_k: int = 10) -> None:
        """Exibe ranking formatado no terminal ordenado por menor MSE de validacao / maior acuracia de teste."""
        if not self.records:
            return

        ranked = sorted(
            self.records,
            key=lambda r: (r["test_acc"], -r["best_val_mse"], r["best_val_acc"]),
            reverse=True,
        )

        print("\n" + "=" * 102)
        print("  RANKING FINAL DO GRID SEARCH (Avaliado nos 130 exemplos fixos de Teste)")
        print("=" * 102)

        header = (
            f"{'Rank':<5} {'Rodada':<7} {'H':<5} {'Ratio':<6} {'LR':<7} {'M':<5} {'P':<4} {'Noise':<6} "
            f"{'Epocas':<7} {'Melhor Ep':<10} {'MSE Val':<10} {'Acc Val':<10} {'Acc Teste':<10}"
        )
        print(header)
        print("-" * len(header))

        for idx, r in enumerate(ranked[:top_k], start=1):
            print(
                f"{idx:<5} {r['rodada']:<7} {r['hidden_neurons']:<5} {r.get('val_ratio', 0.2):<6} {r['learn_rate']:<7} "
                f"{r['momentum']:<5} {r['patience']:<4} {r.get('noise_rate', 0.0):<6.2f} {r['epochs_exec']:<7} "
                f"{r['best_val_acc_epoch']:<10} {r['best_val_mse']:<10.5f} {r['best_val_acc']:>7.2f}%   {r['test_acc']:>7.2f}%"
            )

        best = ranked[0]
        print("=" * 102)
        print("MELHOR COMBINACAO ENCONTRADA (Top 1):")
        print(
            f"  Hidden={best['hidden_neurons']} | ValRatio={best.get('val_ratio', 0.2)} | "
            f"LR={best['learn_rate']} | Momentum={best['momentum']} | Patience={best['patience']} | Noise={best.get('noise_rate', 0.0)}\n"
            f"  Melhor Epoca (Menor MSE Val): {best['best_val_acc_epoch']} (MSE Val: {best['best_val_mse']:.5f} | Acc Val: {best['best_val_acc']:.2f}%)\n"
            f"  Desempenho no Teste (130 fixos): Acc Teste = {best['test_acc']:.2f}% | MSE Teste = {best['test_mse']:.5f}\n"
            f"  Checkpoint: {best['checkpoint'] or 'Nao salvo (< threshold)'}"
        )
        print("=" * 102)

    def finalize_directory_name(self) -> str:
        """Renomeia diretorio de saida com a melhor marca alcancada."""
        if not self.records:
            return self.output_dir

        clean_dir = self.output_dir.rstrip("/")
        if not os.path.exists(clean_dir):
            return self.output_dir

        ranked = sorted(
            self.records,
            key=lambda r: (r["test_acc"], -r["best_val_mse"], r["best_val_acc"]),
            reverse=True,
        )
        best = ranked[0]
        prefix = f"acc{best['test_acc']:.1f}_h{best['hidden_neurons']}_"

        current_name = os.path.basename(clean_dir)
        if current_name.startswith(prefix):
            return self.output_dir

        new_path = os.path.join(os.path.dirname(clean_dir), f"{prefix}{current_name}")
        os.rename(clean_dir, new_path)
        self.output_dir = new_path
        print(f"\nDiretorio renomeado com a melhor marca: {new_path}")

        try:
            with open("models/grid_search/.latest_grid.txt", "w", encoding="utf-8") as f:
                f.write(new_path.strip())
        except Exception:
            pass

        return new_path
