import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


# ==============================================================================
#  Dicionários de Metadados e Rótulos
# ==============================================================================

PARAM_LABELS = {
    "hidden": ("Topologia (H)", lambda v: f"H={int(v)}"),
    "val_ratio": ("Proporcao de Validacao", lambda v: f"{v * 100:.0f}%"),
    "lr": ("Taxa de Aprendizado (η)", lambda v: f"η={v}"),
    "momentum": ("Fator de Momentum (α)", lambda v: f"α={v}"),
    "patience": ("Paciencia (P)", lambda v: f"P={int(v)}"),
    "noise_rate": ("Taxa de Ruido", lambda v: f"{v * 100:.0f}%" if v > 0.0 else "Limpo"),
}

ALL_PARAMS = ["hidden", "val_ratio", "lr", "momentum", "patience", "noise_rate"]


def _get_varied_params(rows: list[dict]) -> list[str]:
    """Retorna os hiperparametros que tiveram mais de um valor na grade."""
    return [p for p in ALL_PARAMS if len(set(r.get(p) for r in rows if p in r)) > 1]


# ==============================================================================
#  1. Painel de Sensibilidade Parametrica Individual (X = Hiperparametro, Y = Acc)
# ==============================================================================

def plot_hyperparam_sensitivity(rows: list[dict], output_path: str) -> str | None:
    """Gera um subplot dedicado para CADA hiperparametro que variou."""
    varied = _get_varied_params(rows)
    if not varied:
        return None

    n = len(varied)
    if n == 1:
        nrows, ncols = 1, 1
        figsize = (8, 5)
    elif n == 2:
        nrows, ncols = 1, 2
        figsize = (14, 5)
    elif n == 3:
        nrows, ncols = 1, 3
        figsize = (18, 5)
    elif n == 4:
        nrows, ncols = 2, 2
        figsize = (14, 10)
    else:  # 5 ou 6
        nrows, ncols = 2, 3
        figsize = (18, 10)

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axes_list = [axs] if n == 1 else (axs.flatten() if hasattr(axs, "flatten") else [axs])

    for idx, p in enumerate(varied):
        ax = axes_list[idx]
        p_name, p_fmt = PARAM_LABELS.get(p, (p, str))

        grouped = defaultdict(list)
        for r in rows:
            grouped[r.get(p)].append(r["best_val_acc"])

        sorted_keys = sorted(grouped.keys())
        labels = [p_fmt(k) for k in sorted_keys]
        data = [grouped[k] for k in sorted_keys]

        # Se cada valor tem 1 amostra: barras; se tem varias: boxplot
        if max((len(v) for v in data), default=0) <= 1:
            vals = [v[0] for v in data]
            bars = ax.bar(labels, vals, color="#3498db", edgecolor="#2980b9", width=0.45)
            for bar in bars:
                h = bar.get_height()
                ax.annotate(
                    f"{h:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontweight="bold",
                    fontsize=9,
                )
            ax.set_ylim(0, max(vals) * 1.15 if vals else 100)
        else:
            ax.boxplot(
                data,
                tick_labels=labels,
                patch_artist=True,
                boxprops=dict(facecolor="#d5dbdb", edgecolor="#34495e"),
                medianprops=dict(color="#e74c3c", linewidth=2),
            )
            # Medias como diamantes azuis
            means = [sum(v) / len(v) for v in data]
            ax.plot(range(1, len(data) + 1), means, "b^", label="Media", markersize=7)
            ax.legend(loc="lower right")

        ax.set_title(f"Sensibilidade: Acuracia vs {p_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel(p_name)
        ax.set_ylabel("Acuracia de Validacao (%)")
        ax.grid(True, linestyle="--", alpha=0.5)

    # Oculta subplots vazios caso haja sobra no grid
    for idx in range(n, len(axes_list)):
        axes_list[idx].axis("off")

    plt.suptitle("Analise de Sensibilidade Individual por Hiperparametro", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Grafico de sensibilidade parametrica salvo em: {output_path}")
    return output_path


# ==============================================================================
#  2. Mapas de Calor 2D (Interação Cruzada entre Hiperparâmetros)
# ==============================================================================

def plot_heatmaps(rows: list[dict], output_path: str) -> str | None:
    """Gera heatmaps 2D de combinacoes teoricas classicas (LR x Momentum, H x LR)."""
    varied = _get_varied_params(rows)
    if len(varied) < 2:
        return None

    # Pares teoricamente mais relevantes em redes neurais
    desired_pairs = [
        ("lr", "momentum"),
        ("hidden", "lr"),
        ("noise_rate", "lr"),
        ("hidden", "momentum"),
    ]

    active_pairs = [pair for pair in desired_pairs if pair[0] in varied and pair[1] in varied]

    # Se nenhum par classico variou junto, pega as duas primeiras variaveis disponiveis
    if not active_pairs:
        active_pairs = [(varied[0], varied[1])]

    # Limita a no maximo 2 heatmaps para manter a imagem limpa e focada
    active_pairs = active_pairs[:2]
    ncols = len(active_pairs)
    fig, axs = plt.subplots(1, ncols, figsize=(8 * ncols, 6))
    axes_list = [axs] if ncols == 1 else axs

    for idx, (px, py) in enumerate(active_pairs):
        ax = axes_list[idx]
        x_name, x_fmt = PARAM_LABELS.get(px, (px, str))
        y_name, y_fmt = PARAM_LABELS.get(py, (py, str))

        vx_list = sorted(list(set(r.get(px) for r in rows)))
        vy_list = sorted(list(set(r.get(py) for r in rows)))

        matrix = np.full((len(vy_list), len(vx_list)), np.nan)
        for i, vy in enumerate(vy_list):
            for j, vx in enumerate(vx_list):
                matches = [r for r in rows if r.get(px) == vx and r.get(py) == vy]
                if matches:
                    matrix[i, j] = max(r["best_val_acc"] for r in matches)

        im = ax.imshow(matrix, cmap="viridis", aspect="auto", origin="lower")
        ax.set_xticks(range(len(vx_list)))
        ax.set_xticklabels([x_fmt(v) for v in vx_list])
        ax.set_yticks(range(len(vy_list)))
        ax.set_yticklabels([y_fmt(v) for v in vy_list])
        ax.set_title(f"Interacao: {y_name} vs {x_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)

        # Rótulos nas células
        t_min, t_max = np.nanmin(matrix), np.nanmax(matrix)
        threshold = t_min + (t_max - t_min) * 0.5
        for i in range(len(vy_list)):
            for j in range(len(vx_list)):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = "black" if val > threshold else "white"
                    ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontweight="bold")

        fig.colorbar(im, ax=ax, label="Maior Acuracia de Validacao (%)")

    plt.suptitle("Mapas de Calor de Interacao Bivariada", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Mapas de calor bivariados salvos em: {output_path}")
    return output_path


# ==============================================================================
#  3. Visão Sistêmica Global (Fronteira MSE, Overfitting Gap, Custo Computacional)
# ==============================================================================

def plot_global_insights(rows: list[dict], output_path: str) -> str:
    """Gera painel com Fronteira MSE vs Acc, Gap de Overfitting e Custo Computacional."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    varied = _get_varied_params(rows)
    primary_param = varied[0] if varied else "hidden"
    p_name, p_fmt = PARAM_LABELS.get(primary_param, (primary_param, str))
    best = max(rows, key=lambda x: (x["best_val_acc"], -x["best_val_mse"]))

    # 1. Fronteira MSE vs Acurácia
    unique_vals = sorted(list(set(r.get(primary_param, 0) for r in rows)))
    for val in unique_vals:
        sub = [r for r in rows if r.get(primary_param) == val]
        axs[0].scatter(
            [r["best_val_mse"] for r in sub],
            [r["best_val_acc"] for r in sub],
            label=p_fmt(val),
            alpha=0.85,
            s=70,
        )

    axs[0].scatter(
        [best["best_val_mse"]],
        [best["best_val_acc"]],
        color="#f39c12",
        edgecolors="black",
        s=160,
        linewidths=2,
        zorder=5,
        label=f"Campeao ({best['best_val_acc']:.2f}%)",
    )
    axs[0].set_title("Fronteira: MSE vs Acuracia de Validacao", fontsize=11, fontweight="bold")
    axs[0].set_xlabel("Menor MSE de Validacao")
    axs[0].set_ylabel("Acuracia de Validacao (%)")
    axs[0].grid(True, linestyle="--", alpha=0.5)
    axs[0].legend()

    # 2. Diagnóstico de Generalização & Overfitting
    sorted_rows = sorted(rows, key=lambda x: x["rodada"])
    n = len(sorted_rows)
    train_accs = [r["final_train_acc"] for r in sorted_rows]
    val_accs = [r["best_val_acc"] for r in sorted_rows]

    if n <= 20:
        idx = np.arange(n)
        width = 0.35
        axs[1].bar(idx - width / 2, train_accs, width, label="Treino Final", color="#2980b9", alpha=0.85)
        axs[1].bar(idx + width / 2, val_accs, width, label="Melhor Validacao", color="#27ae60", alpha=0.85)
        axs[1].set_xticks(idx)
        axs[1].set_xticklabels([f"#{r['rodada']}" for r in sorted_rows])
    else:
        rounds = [r["rodada"] for r in sorted_rows]
        axs[1].plot(rounds, train_accs, label="Treino Final", color="#2980b9", linewidth=2, marker="o", markersize=3)
        axs[1].plot(rounds, val_accs, label="Melhor Validacao", color="#27ae60", linewidth=2, marker="s", markersize=3)
        axs[1].fill_between(rounds, train_accs, val_accs, color="#e67e22", alpha=0.2, label="Gap de Overfitting")

    axs[1].set_title("Diagnostico de Generalizacao: Treino vs Validacao", fontsize=11, fontweight="bold")
    axs[1].set_xlabel("Rodada de Treinamento")
    axs[1].set_ylabel("Acuracia (%)")
    axs[1].grid(True, linestyle="--", alpha=0.5)
    axs[1].legend()

    # 3. Custo Computacional vs Acurácia
    durations = [r["duracao_s"] for r in rows]
    accs = [r["best_val_acc"] for r in rows]
    epochs = [r["epochs_exec"] for r in rows]

    scatter = axs[2].scatter(
        durations,
        accs,
        c=[r.get(primary_param, 0) for r in rows],
        cmap="viridis",
        s=[max(40, ep * 3) for ep in epochs],
        alpha=0.85,
        edgecolors="black",
        linewidths=0.5,
    )
    axs[2].set_title(f"Custo Computacional vs Acuracia ({p_name})", fontsize=11, fontweight="bold")
    axs[2].set_xlabel("Duracao (segundos)")
    axs[2].set_ylabel("Acuracia de Validacao (%)")
    axs[2].grid(True, linestyle="--", alpha=0.5)
    fig.colorbar(scatter, ax=axs[2], label=p_name)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Grafico de visao global salvo em: {output_path}")
    return output_path


# ==============================================================================
#  Orquestrador Geral de Visualizações
# ==============================================================================

def plot_all_grid_visualizations(rows: list[dict], target_dir: str) -> dict[str, str]:
    """Gera o conjunto completo de visualizacoes dedicadas."""
    os.makedirs(target_dir, exist_ok=True)
    generated = {}

    # 1. Visão global sistêmica
    path_insights = os.path.join(target_dir, "grid_search_insights.png")
    generated["insights"] = plot_global_insights(rows, path_insights)

    # 2. Sensibilidade de CADA hiperparâmetro individual
    path_hyper = os.path.join(target_dir, "grid_search_hyperparams.png")
    res_hyper = plot_hyperparam_sensitivity(rows, path_hyper)
    if res_hyper:
        generated["hyperparams"] = res_hyper

    # 3. Heatmaps bivariados (se houver >= 2 variáveis)
    path_heatmaps = os.path.join(target_dir, "grid_search_heatmaps.png")
    res_heat = plot_heatmaps(rows, path_heatmaps)
    if res_heat:
        generated["heatmaps"] = res_heat

    return generated


# Compatibilidade retroativa
def plot_grid_search_insights(rows: list[dict], output_path: str) -> str:
    """Funcao de compatibilidade que redireciona para plot_global_insights."""
    return plot_global_insights(rows, output_path)
