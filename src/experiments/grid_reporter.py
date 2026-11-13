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
        val_ratio: float = 0.1,
        noise_rate: float = 0.0,
        checkpoint_path: str | None = None,
    ) -> dict:
        """Registra metricas da rodada de grid search."""
        duration = getattr(orch, "delta_ts", None)
        if duration is None and hasattr(orch, "start_ts"):
            duration = datetime.now().timestamp() - orch.start_ts
        else:
            duration = duration or 0.0

        val_accs = orch.history.get("val_acc", [])
        train_accs = orch.history.get("train_acc", [])
        val_mses = orch.history.get("val_mse", [])

        best_val_acc = (max(val_accs) * 100) if val_accs else 0.0
        best_val_acc_epoch = (val_accs.index(max(val_accs)) + 1) if val_accs else 0
        best_val_mse = min(val_mses) if val_mses else 0.0

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
            "best_val_acc": round(best_val_acc, 2),
            "best_val_acc_epoch": best_val_acc_epoch,
            "final_val_mse": round(val_mses[-1] if val_mses else 0.0, 5),
            "best_val_mse": round(best_val_mse, 5),
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
        """Exibe ranking formatado no terminal ordenado por acuracia de validacao."""
        if not self.records:
            return

        ranked = sorted(self.records, key=lambda r: (r["best_val_acc"], -r["best_val_mse"]), reverse=True)

        print("\n" + "=" * 88)
        print("  RANKING FINAL DO GRID SEARCH (Ordenado por Maior Acuracia de Validacao)")
        print("=" * 88)

        header = (
            f"{'Rank':<5} {'Rodada':<7} {'H':<5} {'Ratio':<6} {'LR':<7} {'M':<5} {'P':<4} {'Noise':<6} "
            f"{'Epocas':<7} {'Best Val Acc':<14} {'Melhor Epoca':<13} {'Best MSE':<10}"
        )
        print(header)
        print("-" * len(header))

        for idx, r in enumerate(ranked[:top_k], start=1):
            print(
                f"{idx:<5} {r['rodada']:<7} {r['hidden_neurons']:<5} {r.get('val_ratio', 0.1):<6} {r['learn_rate']:<7} "
                f"{r['momentum']:<5} {r['patience']:<4} {r.get('noise_rate', 0.0):<6.2f} {r['epochs_exec']:<7} "
                f"{r['best_val_acc']:>10.2f}%   {r['best_val_acc_epoch']:<13} {r['best_val_mse']:<10.5f}"
            )

        best = ranked[0]
        print("=" * 88)
        print("MELHOR COMBINACAO ENCONTRADA (Top 1):")
        print(
            f"  Hidden={best['hidden_neurons']} | Ratio={best.get('val_ratio', 0.1)} | "
            f"LR={best['learn_rate']} | Momentum={best['momentum']} | Patience={best['patience']} | Noise={best.get('noise_rate', 0.0)}\n"
            f"  Acuracia de Validacao: {best['best_val_acc']:.2f}% (alcancada na epoca {best['best_val_acc_epoch']})\n"
            f"  MSE de Validacao: {best['best_val_mse']:.5f}\n"
            f"  Checkpoint: {best['checkpoint'] or 'Nao salvo (< threshold)'}"
        )
        print("=" * 88)

    def finalize_directory_name(self) -> str:
        """Renomeia diretorio de saida com a melhor marca alcancada."""
        if not self.records:
            return self.output_dir

        clean_dir = self.output_dir.rstrip("/")
        if not os.path.exists(clean_dir):
            return self.output_dir

        ranked = sorted(self.records, key=lambda r: (r["best_val_acc"], -r["best_val_mse"]), reverse=True)
        best = ranked[0]
        prefix = f"acc{best['best_val_acc']:.1f}_h{best['hidden_neurons']}_"

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
