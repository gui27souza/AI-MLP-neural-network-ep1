import csv
import os
import sys
from collections import defaultdict

from lab.analysis.grid_plots import plot_grid_search_insights


class GridSearchEvaluator:
    """Orquestrador de relatorios analiticos e visualizacoes de Grid Search."""

    def __init__(self, target_dir: str):
        if not os.path.exists(target_dir):
            raise FileNotFoundError(f"Diretorio de Grid Search nao encontrado: {target_dir}")

        self.target_dir = target_dir.rstrip("/\\")
        self.summary_csv = os.path.join(self.target_dir, "summary.csv")
        if not os.path.exists(self.summary_csv):
            raise FileNotFoundError(f"Arquivo summary.csv nao encontrado em: {self.target_dir}")

        self.rows = self._load_rows()

    def _load_rows(self) -> list[dict]:
        rows = []
        with open(self.summary_csv, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({
                    "rodada": int(r["rodada"]),
                    "hidden": int(r["hidden_neurons"]),
                    "lr": float(r["learn_rate"]),
                    "momentum": float(r["momentum"]),
                    "patience": int(r["patience"]),
                    "noise_rate": float(r["noise_rate"]) if "noise_rate" in r and r["noise_rate"] else 0.0,
                    "val_ratio": float(r["val_ratio"]),
                    "epochs_exec": int(r["epochs_exec"]),
                    "stopped_early": r["stopped_early"].lower() == "true",
                    "best_val_acc": float(r["best_val_acc"]),
                    "best_val_mse": float(r["best_val_mse"]),
                    "duracao_s": float(r["duracao_s"]),
                    "checkpoint": r["checkpoint"],
                })
        return rows

    def generate_plots(self, output_png: str | None = None) -> str:
        dest_png = output_png or os.path.join(self.target_dir, "grid_search_insights.png")
        return plot_grid_search_insights(self.rows, dest_png)

    def generate_report(self, output_md: str | None = None) -> str:
        dest_md = output_md or os.path.join(self.target_dir, "ANALISE_GRID_SEARCH.md")
        sorted_rows = sorted(self.rows, key=lambda x: (-x["best_val_acc"], x["best_val_mse"]))
        best = sorted_rows[0]

        acc_by_h = defaultdict(list)
        time_by_h = defaultdict(list)
        for r in self.rows:
            acc_by_h[r["hidden"]].append(r["best_val_acc"])
            time_by_h[r["hidden"]].append(r["duracao_s"])

        md = [
            f"# Relatorio Analitico - Grid Search\n\nTotal de Modelos Treinados: {len(self.rows)}\n",
            "## 1. Melhor Modelo da Execucao\n",
            f"- **Rodada:** #{best['rodada']}",
            f"- **Topologia:** `H = {best['hidden']}`",
            f"- **Hiperparametros:** `LR = {best['lr']}`, `Momentum = {best['momentum']}`, `Patience = {best['patience']}`, `Noise = {best.get('noise_rate', 0.0)}`",
            f"- **Melhor Acuracia de Validacao:** **{best['best_val_acc']:.2f}%**",
            f"- **Menor MSE de Validacao:** **{best['best_val_mse']:.5f}**",
            f"- **Epocas Executadas:** {best['epochs_exec']} ({best['duracao_s']:.2f}s)",
            f"- **Checkpoint Salvo:** `{best['checkpoint'] or 'Descartado por limiar de acuracia'}`\n",
            "## 2. Comparativo por Topologia (H)\n",
            "| Neuronios (H) | Total Modelos | Acuracia Media | Acuracia Maxima | MSE Minimo | Tempo Medio (s) |",
            "| :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for h in sorted(acc_by_h.keys()):
            mean_acc = sum(acc_by_h[h]) / len(acc_by_h[h])
            max_acc = max(acc_by_h[h])
            min_mse = min(r["best_val_mse"] for r in self.rows if r["hidden"] == h)
            mean_time = sum(time_by_h[h]) / len(time_by_h[h])
            md.append(f"| **H={h}** | {len(acc_by_h[h])} | {mean_acc:.2f}% | **{max_acc:.2f}%** | **{min_mse:.5f}** | {mean_time:.2f}s |")

        # Comparativo por Ruído (se houver variação de ruído ou ruído ativo)
        noise_vals = sorted(list(set(r.get("noise_rate", 0.0) for r in self.rows)))
        if len(noise_vals) > 1 or (len(noise_vals) == 1 and noise_vals[0] > 0.0):
            md.extend([
                "\n## 3. Comparativo por Taxa de Ruido (Data Augmentation)\n",
                "| Taxa de Ruido | Total Modelos | Acuracia Media | Acuracia Maxima | MSE Minimo | Tempo Medio (s) |",
                "| :---: | :---: | :---: | :---: | :---: | :---: |",
            ])
            for n_rate in noise_vals:
                sub_noise = [r for r in self.rows if r.get("noise_rate", 0.0) == n_rate]
                mean_acc_n = sum(r["best_val_acc"] for r in sub_noise) / len(sub_noise)
                max_acc_n = max(r["best_val_acc"] for r in sub_noise)
                min_mse_n = min(r["best_val_mse"] for r in sub_noise)
                mean_time_n = sum(r["duracao_s"] for r in sub_noise) / len(sub_noise)
                label_n = f"{n_rate * 100:.1f}%" if n_rate > 0 else "0.0% (Limpo)"
                md.append(f"| **{label_n}** | {len(sub_noise)} | {mean_acc_n:.2f}% | **{max_acc_n:.2f}%** | **{min_mse_n:.5f}** | {mean_time_n:.2f}s |")

        section_num = 4 if (len(noise_vals) > 1 or (len(noise_vals) == 1 and noise_vals[0] > 0.0)) else 3
        md.extend([
            f"\n## {section_num}. Top 5 Melhores Modelos\n",
            "| Posicao | Rodada | H | LR | Momentum | Noise | Acc Val | MSE Val | Epocas | Tempo | Checkpoint |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        ])

        for rank, r in enumerate(sorted_rows[:5], start=1):
            md.append(
                f"| #{rank} | Rodada {r['rodada']} | H={r['hidden']} | {r['lr']} | {r['momentum']} | {r.get('noise_rate', 0.0):.2f} | "
                f"**{r['best_val_acc']:.2f}%** | {r['best_val_mse']:.5f} | {r['epochs_exec']} | {r['duracao_s']:.1f}s | `{r['checkpoint']}` |"
            )

        with open(dest_md, "w", encoding="utf-8") as f:
            f.write("\n".join(md) + "\n")

        print(f"Relatorio analitico salvo em: {dest_md}")
        return dest_md


if __name__ == "__main__":
    grid_dir = os.environ.get("GRID_DIR") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not grid_dir:
        print("\nErro: Nenhum diretorio de grid search informado.")
        print("Uso: make evaluate_grid GRID_DIR=models/grid_search/...\n")
        sys.exit(1)

    evaluator = GridSearchEvaluator(grid_dir)
    evaluator.generate_plots()
    evaluator.generate_report()
