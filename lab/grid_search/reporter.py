import csv
import os
from datetime import datetime

from src.orchestrator import Orchestrator


class GridSearchReporter:
    """Responsável por consolidar métricas, exportar CSV e exibir rankings do Grid Search."""

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
        checkpoint_path: str | None = None,
    ) -> dict:
        """Extrai métricas do orquestrador e armazena o resultado da rodada."""

        duration = getattr(orch, "delta_ts", None)
        if duration is None and hasattr(orch, "start_ts"):
            duration = datetime.now().timestamp() - orch.start_ts
        else:
            duration = duration or 0.0

        best_val_acc = (max(orch.history["val_acc"]) * 100) if orch.history["val_acc"] else 0.0
        best_val_acc_epoch = (orch.history["val_acc"].index(max(orch.history["val_acc"])) + 1) if orch.history["val_acc"] else 0
        best_val_mse = min(orch.history["val_mse"]) if orch.history["val_mse"] else 0.0

        final_val_acc = (orch.history["val_acc"][-1] * 100) if orch.history["val_acc"] else 0.0
        final_train_acc = (orch.history["train_acc"][-1] * 100) if orch.history["train_acc"] else 0.0
        final_val_mse = orch.history["val_mse"][-1] if orch.history["val_mse"] else 0.0

        checkpoint_filename = os.path.basename(checkpoint_path) if checkpoint_path else ""

        record = {
            "rodada": round_num,
            "hidden_neurons": hidden_neurons,
            "val_ratio": val_ratio,
            "learn_rate": learn_rate,
            "momentum": momentum,
            "patience": patience,
            "epochs_exec": orch.total_epochs,
            "stopped_early": orch.stopped_early,
            "final_train_acc": round(final_train_acc, 2),
            "final_val_acc": round(final_val_acc, 2),
            "best_val_acc": round(best_val_acc, 2),
            "best_val_acc_epoch": best_val_acc_epoch,
            "final_val_mse": round(final_val_mse, 5),
            "best_val_mse": round(best_val_mse, 5),
            "duracao_s": round(duration, 2),
            "checkpoint": checkpoint_filename,
        }

        self.records.append(record)
        return record

    def save_csv(self, filename: str = "summary.csv") -> str:
        """Salva todos os registros em arquivo CSV no diretório de saída."""

        if not self.records:
            print("Nenhum registro encontrado para salvar o CSV.")
            return ""

        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = os.path.join(self.output_dir, filename)

        fieldnames = list(self.records[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

        print(f"\nResumo consolidado salvo em: {csv_path}")
        return csv_path

    def print_ranking(self, top_k: int = 10) -> None:
        """Exibe uma tabela formatada no terminal com os melhores modelos por acurácia de validação."""

        if not self.records:
            print("Nenhum registro para exibir ranking.")
            return

        print("\n" + "=" * 88)
        print("  RANKING FINAL DO GRID SEARCH (Ordenado por Maior Acuracia de Validacao)")
        print("=" * 88)

        # Ordenação decrescente pela melhor acurácia de validação e desempate pelo menor MSE
        ranked = sorted(
            self.records,
            key=lambda r: (r["best_val_acc"], -r["best_val_mse"]),
            reverse=True,
        )

        header = (
            f"{'Rank':<5} {'Rodada':<7} {'H':<5} {'Ratio':<6} {'LR':<7} {'M':<5} {'P':<4} "
            f"{'Epocas':<7} {'Best Val Acc':<14} {'Melhor Epoca':<13} {'Best MSE':<10}"
        )
        print(header)
        print("-" * len(header))

        for idx, r in enumerate(ranked[:top_k], start=1):
            ratio_val = r.get("val_ratio", 0.1)
            print(
                f"{idx:<5} {r['rodada']:<7} {r['hidden_neurons']:<5} {ratio_val:<6} {r['learn_rate']:<7} "
                f"{r['momentum']:<5} {r['patience']:<4} {r['epochs_exec']:<7} "
                f"{r['best_val_acc']:>10.2f}%   {r['best_val_acc_epoch']:<13} {r['best_val_mse']:<10.5f}"
            )

        best = ranked[0]
        ratio_best = best.get("val_ratio", 0.1)
        print("=" * 88)
        print("MELHOR COMBINACAO ENCONTRADA (Top 1):")
        print(
            f"  Hidden={best['hidden_neurons']} | Ratio={ratio_best} | LR={best['learn_rate']} | Momentum={best['momentum']} | Patience={best['patience']}\n"
            f"  Acuracia de Validacao: {best['best_val_acc']:.2f}% (alcancada na epoca {best['best_val_acc_epoch']})\n"
            f"  MSE de Validacao: {best['best_val_mse']:.5f}\n"
            f"  Checkpoint: {best['checkpoint']}"
        )
        print("=" * 88)

    def finalize_directory_name(self) -> str:
        """Renomeia o diretorio de saida para incluir a melhor arquitetura e acuracia alcancadas."""

        if not self.records:
            return self.output_dir

        clean_dir = self.output_dir.rstrip("/")
        if not os.path.exists(clean_dir):
            return self.output_dir

        ranked = sorted(
            self.records,
            key=lambda r: (r["best_val_acc"], -r["best_val_mse"]),
            reverse=True,
        )
        best = ranked[0]
        best_h = best["hidden_neurons"]
        best_acc = best["best_val_acc"]

        parent_dir = os.path.dirname(clean_dir)
        current_name = os.path.basename(clean_dir)

        prefix = f"h{best_h}_acc{best_acc:.1f}_"
        if current_name.startswith(prefix):
            return self.output_dir

        new_name = f"{prefix}{current_name}"
        new_path = os.path.join(parent_dir, new_name)

        os.rename(clean_dir, new_path)
        self.output_dir = new_path
        print(f"\nDiretorio renomeado com a melhor marca: {new_path}")
        return new_path
