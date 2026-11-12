import csv
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt


class GridSearchEvaluator:
    """Orquestrador de analise de rodadas de Grid Search: relatorios em Markdown e painel visual em PNG."""

    def __init__(self, target_dir: str):
        if not os.path.exists(target_dir):
            raise FileNotFoundError(f"Diretorio de Grid Search nao encontrado: {target_dir}")

        self.target_dir = target_dir.rstrip("/\\")
        self.summary_csv = os.path.join(self.target_dir, "summary.csv")
        if not os.path.exists(self.summary_csv):
            raise FileNotFoundError(f"Arquivo summary.csv nao encontrado em: {self.target_dir}")

        self.rows: list[dict] = []
        with open(self.summary_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.rows.append({
                    "rodada": int(r["rodada"]),
                    "hidden": int(r["hidden_neurons"]),
                    "lr": float(r["learn_rate"]),
                    "momentum": float(r["momentum"]),
                    "patience": int(r["patience"]),
                    "val_ratio": float(r["val_ratio"]),
                    "epochs_exec": int(r["epochs_exec"]),
                    "stopped_early": r["stopped_early"].lower() == "true",
                    "best_val_acc": float(r["best_val_acc"]),
                    "best_val_mse": float(r["best_val_mse"]),
                    "duracao_s": float(r["duracao_s"]),
                    "checkpoint": r["checkpoint"],
                })

    def generate_plots(self, output_png: str | None = None) -> str:
        dest_png = output_png or os.path.join(self.target_dir, "grid_search_insights.png")

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Dispersao MSE vs Acuracia
        hidden_vals = sorted(list(set(r["hidden"] for r in self.rows)))
        for h in hidden_vals:
            sub = [r for r in self.rows if r["hidden"] == h]
            mses = [r["best_val_mse"] for r in sub]
            accs = [r["best_val_acc"] for r in sub]
            axs[0].scatter(mses, accs, label=f"H={h}", alpha=0.85, s=70)

        # Destaca o campeao
        best_row = max(self.rows, key=lambda x: (x["best_val_acc"], -x["best_val_mse"]))
        axs[0].scatter(
            [best_row["best_val_mse"]],
            [best_row["best_val_acc"]],
            color="#f39c12",
            edgecolors="black",
            s=160,
            linewidths=2,
            zorder=5,
            label=f"Campeao ({best_row['best_val_acc']:.2f}%)",
        )
        axs[0].set_title("Fronteira: MSE vs Acuracia de Validacao", fontsize=11, fontweight="bold")
        axs[0].set_xlabel("Menor MSE de Validacao")
        axs[0].set_ylabel("Acuracia de Validacao (%)")
        axs[0].grid(True, linestyle="--", alpha=0.5)
        axs[0].legend()

        # 2. Boxplot de Acuracia por Topologia
        acc_by_h = defaultdict(list)
        for r in self.rows:
            acc_by_h[r["hidden"]].append(r["best_val_acc"])
        labels = [f"H={h}" for h in sorted(acc_by_h.keys())]
        data = [acc_by_h[h] for h in sorted(acc_by_h.keys())]

        axs[1].boxplot(data, tick_labels=labels, patch_artist=True, boxprops=dict(facecolor="#d5dbdb"))
        axs[1].set_title("Distribuicao de Acuracia por Topologia", fontsize=11, fontweight="bold")
        axs[1].set_ylabel("Acuracia de Validacao (%)")
        axs[1].grid(True, linestyle="--", alpha=0.5)

        # 3. Tempo vs Acuracia
        durations = [r["duracao_s"] for r in self.rows]
        accs = [r["best_val_acc"] for r in self.rows]
        scatter = axs[2].scatter(durations, accs, c=[r["hidden"] for r in self.rows], cmap="viridis", s=70, alpha=0.85)
        axs[2].set_title("Custo Computacional vs Acuracia", fontsize=11, fontweight="bold")
        axs[2].set_xlabel("Duracao (segundos)")
        axs[2].set_ylabel("Acuracia de Validacao (%)")
        axs[2].grid(True, linestyle="--", alpha=0.5)
        fig.colorbar(scatter, ax=axs[2], label="H (neuronios)")

        plt.tight_layout()
        plt.savefig(dest_png, dpi=200)
        plt.close()
        print(f"Graficos analiticos salvos em: {dest_png}")
        return dest_png

    def generate_report(self, output_md: str | None = None) -> str:
        dest_md = output_md or os.path.join(self.target_dir, "ANALISE_GRID_SEARCH.md")
        sorted_rows = sorted(self.rows, key=lambda x: (-x["best_val_acc"], x["best_val_mse"]))
        best = sorted_rows[0]

        acc_by_h = defaultdict(list)
        time_by_h = defaultdict(list)
        for r in self.rows:
            acc_by_h[r["hidden"]].append(r["best_val_acc"])
            time_by_h[r["hidden"]].append(r["duracao_s"])

        md_content = f"""# Relatorio Analitico - Grid Search

Total de Modelos Treinados: {len(self.rows)}  

---

## 1. Melhor Modelo da Execucao

- **Rodada:** #{best['rodada']}
- **Topologia:** `H = {best['hidden']}`
- **Hiperparametros:** `LR = {best['lr']}`, `Momentum = {best['momentum']}`, `Patience = {best['patience']}`
- **Melhor Acuracia de Validacao:** **{best['best_val_acc']:.2f}%**
- **Menor MSE de Validacao:** **{best['best_val_mse']:.5f}**
- **Epocas Executadas:** {best['epochs_exec']} (convergencia em {best['duracao_s']:.2f}s)
- **Checkpoint Salvo:** `{best['checkpoint'] if best['checkpoint'] else 'Descartado por limiar de acuracia'}`

---

## 2. Comparativo por Topologia (H)

| Neuronios (H) | Total Modelos | Acuracia Media | Acuracia Maxima | MSE Minimo | Tempo Medio (s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
"""
        for h in sorted(acc_by_h.keys()):
            mean_acc = sum(acc_by_h[h]) / len(acc_by_h[h])
            max_acc = max(acc_by_h[h])
            min_mse = min(r["best_val_mse"] for r in self.rows if r["hidden"] == h)
            mean_time = sum(time_by_h[h]) / len(time_by_h[h])
            md_content += (
                f"| **H={h}** | {len(acc_by_h[h])} | {mean_acc:.2f}% | "
                f"**{max_acc:.2f}%** | **{min_mse:.5f}** | {mean_time:.2f}s |\n"
            )

        md_content += """
---

## 3. Top 5 Melhores Modelos

| Posicao | Rodada | H | LR | Momentum | Acc Val | MSE Val | Epocas | Tempo | Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
"""
        for rank, r in enumerate(sorted_rows[:5], start=1):
            md_content += (
                f"| #{rank} | Rodada {r['rodada']} | H={r['hidden']} | {r['lr']} | {r['momentum']} | "
                f"**{r['best_val_acc']:.2f}%** | {r['best_val_mse']:.5f} | {r['epochs_exec']} | {r['duracao_s']:.1f}s | "
                f"`{r['checkpoint']}` |\n"
            )

        with open(dest_md, "w", encoding="utf-8") as f:
            f.write(md_content)

        print(f"Relatorio analitico salvo em: {dest_md}")
        return dest_md


if __name__ == "__main__":
    grid_dir = os.environ.get("GRID_DIR")
    if not grid_dir and len(sys.argv) > 1:
        grid_dir = sys.argv[1]

    if not grid_dir:
        print("\nErro: Nenhum diretorio de grid search informado.")
        print("Uso: make evaluate_grid GRID_DIR=models/grid_search/...")
        print(" ou: python3 -m lab.grid_evaluator caminho/do/diretorio\n")
        sys.exit(1)

    evaluator = GridSearchEvaluator(grid_dir)
    evaluator.generate_plots()
    evaluator.generate_report()
