import os
from collections import defaultdict
import matplotlib.pyplot as plt


def plot_grid_search_insights(rows: list[dict], output_path: str) -> str:
    """Gera painel visual com dispersao MSE vs Acc, boxplots e custo computacional."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Dispersao MSE vs Acuracia
    hidden_vals = sorted(list(set(r["hidden"] for r in rows)))
    for h in hidden_vals:
        sub = [r for r in rows if r["hidden"] == h]
        axs[0].scatter([r["best_val_mse"] for r in sub], [r["best_val_acc"] for r in sub], label=f"H={h}", alpha=0.85, s=70)

    best = max(rows, key=lambda x: (x["best_val_acc"], -x["best_val_mse"]))
    axs[0].scatter(
        [best["best_val_mse"]], [best["best_val_acc"]],
        color="#f39c12", edgecolors="black", s=160, linewidths=2, zorder=5,
        label=f"Campeao ({best['best_val_acc']:.2f}%)",
    )
    axs[0].set_title("Fronteira: MSE vs Acuracia de Validacao", fontsize=11, fontweight="bold")
    axs[0].set_xlabel("Menor MSE de Validacao")
    axs[0].set_ylabel("Acuracia de Validacao (%)")
    axs[0].grid(True, linestyle="--", alpha=0.5)
    axs[0].legend()

    # 2. Boxplot de Acuracia por Topologia
    acc_by_h = defaultdict(list)
    for r in rows:
        acc_by_h[r["hidden"]].append(r["best_val_acc"])
    labels = [f"H={h}" for h in sorted(acc_by_h.keys())]
    data = [acc_by_h[h] for h in sorted(acc_by_h.keys())]
    axs[1].boxplot(data, tick_labels=labels, patch_artist=True, boxprops=dict(facecolor="#d5dbdb"))
    axs[1].set_title("Distribuicao de Acuracia por Topologia", fontsize=11, fontweight="bold")
    axs[1].set_ylabel("Acuracia de Validacao (%)")
    axs[1].grid(True, linestyle="--", alpha=0.5)

    # 3. Tempo vs Acuracia
    durations = [r["duracao_s"] for r in rows]
    accs = [r["best_val_acc"] for r in rows]
    scatter = axs[2].scatter(durations, accs, c=[r["hidden"] for r in rows], cmap="viridis", s=70, alpha=0.85)
    axs[2].set_title("Custo Computacional vs Acuracia", fontsize=11, fontweight="bold")
    axs[2].set_xlabel("Duracao (segundos)")
    axs[2].set_ylabel("Acuracia de Validacao (%)")
    axs[2].grid(True, linestyle="--", alpha=0.5)
    fig.colorbar(scatter, ax=axs[2], label="H (neuronios)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Graficos analiticos salvos em: {output_path}")
    return output_path
