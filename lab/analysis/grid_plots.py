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


def _get_test_acc(r: dict) -> float:
    return float(r.get("test_acc", r.get("best_val_acc", 0.0)))


def _get_test_mse(r: dict) -> float:
    return float(r.get("test_mse", r.get("best_val_mse", 0.0)))


# ==============================================================================
#  1. Painel de Sensibilidade Parametrica Individual (Validação vs. Teste)
# ==============================================================================

def plot_hyperparam_sensitivity(rows: list[dict], output_path: str) -> str | None:
    """Gera um subplot dedicado para CADA hiperparametro comparando Validacao e Teste."""
    varied = _get_varied_params(rows)
    if not varied:
        return None

    n = len(varied)
    if n == 1:
        nrows, ncols, figsize = 1, 1, (9, 5.5)
    elif n == 2:
        nrows, ncols, figsize = 1, 2, (15, 5.5)
    elif n == 3:
        nrows, ncols, figsize = 1, 3, (19, 5.5)
    elif n == 4:
        nrows, ncols, figsize = 2, 2, (15, 10.5)
    else:
        nrows, ncols, figsize = 2, 3, (19, 10.5)

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axes_list = [axs] if n == 1 else (axs.flatten() if hasattr(axs, "flatten") else [axs])

    for idx, p in enumerate(varied):
        ax = axes_list[idx]
        p_name, p_fmt = PARAM_LABELS.get(p, (p, str))

        grouped_val = defaultdict(list)
        grouped_test = defaultdict(list)
        for r in rows:
            key = r.get(p)
            grouped_val[key].append(r["best_val_acc"])
            grouped_test[key].append(_get_test_acc(r))

        sorted_keys = sorted(grouped_val.keys())
        labels = [p_fmt(k) for k in sorted_keys]
        data_val = [grouped_val[k] for k in sorted_keys]
        data_test = [grouped_test[k] for k in sorted_keys]
        x_pos = np.arange(len(sorted_keys))

        # Se cada valor tem 1 amostra: barras duplas (Validação vs Teste)
        if max((len(v) for v in data_val), default=0) <= 1:
            vals_v = [v[0] for v in data_val]
            vals_t = [v[0] for v in data_test]
            width = 0.36
            bars_v = ax.bar(
                x_pos - width / 2, vals_v, width=width,
                color="#3498db", edgecolor="#2980b9", label="Acc Validacao", alpha=0.9
            )
            bars_t = ax.bar(
                x_pos + width / 2, vals_t, width=width,
                color="#e67e22", edgecolor="#d35400", label="Acc Teste (130 fixos)", alpha=0.9
            )
            for bar in list(bars_v) + list(bars_t):
                h = bar.get_height()
                ax.annotate(
                    f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontweight="bold",
                    fontsize=8,
                )
            all_vals = vals_v + vals_t
            if all_vals:
                ax.set_ylim(max(0.0, min(all_vals) - 8.0), min(100.0, max(all_vals) + 6.0))
        else:
            # Boxplots pareados: Validação (azul) à esquerda, Teste (laranja) à direita
            offset = 0.18
            bp_val = ax.boxplot(
                data_val,
                positions=x_pos - offset,
                widths=0.30,
                patch_artist=True,
                boxprops=dict(facecolor="#aed6f1", edgecolor="#2980b9", alpha=0.85),
                medianprops=dict(color="#1b4f72", linewidth=2),
                flierprops=dict(marker="o", markersize=4, markerfacecolor="#2980b9"),
            )
            bp_test = ax.boxplot(
                data_test,
                positions=x_pos + offset,
                widths=0.30,
                patch_artist=True,
                boxprops=dict(facecolor="#f5cba7", edgecolor="#d35400", alpha=0.85),
                medianprops=dict(color="#873600", linewidth=2),
                flierprops=dict(marker="o", markersize=4, markerfacecolor="#d35400"),
            )
            means_val = [sum(v) / len(v) for v in data_val]
            means_test = [sum(v) / len(v) for v in data_test]
            ax.plot(
                x_pos - offset, means_val, "b^--",
                label="Media Validacao", markersize=6, linewidth=1.2, alpha=0.85
            )
            ax.plot(
                x_pos + offset, means_test, "r*--",
                label="Media Teste (130 fixos)", markersize=8, linewidth=1.2, alpha=0.9
            )
            # Anota a média de teste acima de cada grupo
            for i, m_t in enumerate(means_test):
                ax.annotate(
                    f"{m_t:.1f}%",
                    xy=(x_pos[i] + offset, max(data_test[i])),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                    color="#873600",
                )
            all_flat = [x for sub in (data_val + data_test) for x in sub]
            if all_flat:
                ax.set_ylim(max(0.0, min(all_flat) - 5.0), min(100.0, max(all_flat) + 6.0))

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.set_title(f"Sensibilidade: {p_name} (Val vs Teste)", fontsize=11, fontweight="bold")
        ax.set_xlabel(p_name)
        ax.set_ylabel("Acuracia (%)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", fontsize=8)

    for idx in range(n, len(axes_list)):
        axes_list[idx].axis("off")

    plt.suptitle(
        "Analise de Sensibilidade por Hiperparametro (Validacao vs Teste Oficial 130 Amostras)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Grafico de sensibilidade parametrica salvo em: {output_path}")
    return output_path


# ==============================================================================
#  2. Mapas de Calor 2D (Interação Cruzada: Teste + Validação)
# ==============================================================================

def plot_heatmaps(rows: list[dict], output_path: str) -> str | None:
    """Gera heatmaps 2D exibindo Acuracia de Teste (principal) e Validacao em cada celula."""
    varied = _get_varied_params(rows)
    if len(varied) < 2:
        return None

    desired_pairs = [
        ("lr", "momentum"),
        ("hidden", "lr"),
        ("noise_rate", "lr"),
        ("hidden", "momentum"),
        ("val_ratio", "lr"),
    ]

    active_pairs = [pair for pair in desired_pairs if pair[0] in varied and pair[1] in varied]
    if not active_pairs:
        active_pairs = [(varied[0], varied[1])]

    active_pairs = active_pairs[:2]

    # Se apenas 1 par variou, plota lado a lado: Heatmap de Teste (esq) e Heatmap de Validação (dir)
    if len(active_pairs) == 1:
        px, py = active_pairs[0]
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        _render_single_heatmap(axs[0], rows, px, py, metric="test_acc", title_prefix="Teste (130 fixos)", cmap="YlGnBu")
        _render_single_heatmap(axs[1], rows, px, py, metric="best_val_acc", title_prefix="Validacao", cmap="viridis")
    else:
        ncols = len(active_pairs)
        fig, axs = plt.subplots(1, ncols, figsize=(8 * ncols, 6))
        axes_list = [axs] if ncols == 1 else axs
        for idx, (px, py) in enumerate(active_pairs):
            _render_single_heatmap(
                axes_list[idx], rows, px, py,
                metric="test_acc", title_prefix="Teste & Val", cmap="viridis", show_both_in_cell=True
            )

    plt.suptitle(
        "Mapas de Calor de Interacao Bivariada (Acuracia de Teste e Validacao)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Mapas de calor bivariados salvos em: {output_path}")
    return output_path


def _render_single_heatmap(
    ax,
    rows: list[dict],
    px: str,
    py: str,
    metric: str = "test_acc",
    title_prefix: str = "Teste",
    cmap: str = "viridis",
    show_both_in_cell: bool = True,
) -> None:
    x_name, x_fmt = PARAM_LABELS.get(px, (px, str))
    y_name, y_fmt = PARAM_LABELS.get(py, (py, str))

    vx_list = sorted(list(set(r.get(px) for r in rows)))
    vy_list = sorted(list(set(r.get(py) for r in rows)))

    matrix_main = np.full((len(vy_list), len(vx_list)), np.nan)
    matrix_val = np.full((len(vy_list), len(vx_list)), np.nan)
    matrix_test = np.full((len(vy_list), len(vx_list)), np.nan)

    for i, vy in enumerate(vy_list):
        for j, vx in enumerate(vx_list):
            matches = [r for r in rows if r.get(px) == vx and r.get(py) == vy]
            if matches:
                best_match = max(matches, key=lambda r: (_get_test_acc(r), -r["best_val_mse"]))
                matrix_test[i, j] = _get_test_acc(best_match)
                matrix_val[i, j] = best_match["best_val_acc"]
                matrix_main[i, j] = matrix_test[i, j] if metric == "test_acc" else max(r["best_val_acc"] for r in matches)

    im = ax.imshow(matrix_main, cmap=cmap, aspect="auto", origin="lower")
    ax.set_xticks(range(len(vx_list)))
    ax.set_xticklabels([x_fmt(v) for v in vx_list])
    ax.set_yticks(range(len(vy_list)))
    ax.set_yticklabels([y_fmt(v) for v in vy_list])
    ax.set_title(f"{title_prefix}: {y_name} vs {x_name}", fontsize=11, fontweight="bold")
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)

    t_min, t_max = np.nanmin(matrix_main), np.nanmax(matrix_main)
    threshold = t_min + (t_max - t_min) * 0.5
    for i in range(len(vy_list)):
        for j in range(len(vx_list)):
            val = matrix_main[i, j]
            if not np.isnan(val):
                # Em YlGnBu valores altos são escuros; em viridis valores altos são claros
                if cmap == "YlGnBu":
                    color = "white" if val > threshold else "black"
                else:
                    color = "black" if val > threshold else "white"

                if show_both_in_cell:
                    cell_text = f"T: {matrix_test[i, j]:.1f}%\n(V: {matrix_val[i, j]:.1f}%)"
                else:
                    cell_text = f"{val:.1f}%"
                ax.text(j, i, cell_text, ha="center", va="center", color=color, fontweight="bold", fontsize=9)

    cbar_label = "Maior Acuracia de Teste (%)" if metric == "test_acc" else "Maior Acuracia de Validacao (%)"
    plt.colorbar(im, ax=ax, label=cbar_label)


# ==============================================================================
#  3. Visão Sistêmica Global (Fronteira MSE Teste/Val, Funil 3 Níveis, Custo vs Teste)
# ==============================================================================

def plot_global_insights(rows: list[dict], output_path: str) -> str:
    """Gera painel com Fronteira MSE vs Acc Teste, Funil Treino/Val/Teste e Custo vs Acc Teste."""
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.5))

    varied = _get_varied_params(rows)
    primary_param = varied[0] if varied else "hidden"
    p_name, p_fmt = PARAM_LABELS.get(primary_param, (primary_param, str))
    best_test = max(rows, key=lambda x: (_get_test_acc(x), -x["best_val_mse"], x["best_val_acc"]))

    # 1. Fronteira MSE de Teste vs Acurácia de Teste (colorido pelo parâmetro principal)
    unique_vals = sorted(list(set(r.get(primary_param, 0) for r in rows)))
    for val in unique_vals:
        sub = [r for r in rows if r.get(primary_param) == val]
        axs[0].scatter(
            [_get_test_mse(r) for r in sub],
            [_get_test_acc(r) for r in sub],
            label=f"{p_fmt(val)} (Teste)",
            alpha=0.85,
            s=75,
            edgecolors="black",
            linewidths=0.5,
        )

    axs[0].scatter(
        [_get_test_mse(best_test)],
        [_get_test_acc(best_test)],
        color="#f1c40f",
        edgecolors="black",
        marker="*",
        s=260,
        linewidths=1.5,
        zorder=6,
        label=f"Campeao Teste ({_get_test_acc(best_test):.2f}%)",
    )
    axs[0].set_title("Fronteira no Teste: MSE vs Acuracia (130 fixos)", fontsize=11, fontweight="bold")
    axs[0].set_xlabel("MSE no Conjunto de Teste")
    axs[0].set_ylabel("Acuracia no Teste (%)")
    axs[0].grid(True, linestyle="--", alpha=0.5)
    axs[0].legend(fontsize=8)

    # 2. Funil de Generalização: Treino -> Validação -> Teste
    sorted_rows = sorted(rows, key=lambda x: x["rodada"])
    n = len(sorted_rows)
    train_accs = [r["final_train_acc"] for r in sorted_rows]
    val_accs = [r["best_val_acc"] for r in sorted_rows]
    test_accs = [_get_test_acc(r) for r in sorted_rows]

    if n <= 15:
        idx = np.arange(n)
        width = 0.27
        axs[1].bar(idx - width, train_accs, width, label="Treino Final", color="#2980b9", alpha=0.85)
        axs[1].bar(idx, val_accs, width, label="Melhor Validacao", color="#27ae60", alpha=0.85)
        axs[1].bar(idx + width, test_accs, width, label="Teste (130 fixos)", color="#e74c3c", alpha=0.9)
        axs[1].set_xticks(idx)
        axs[1].set_xticklabels([f"#{r['rodada']}" for r in sorted_rows], fontsize=8)
    else:
        rounds = [r["rodada"] for r in sorted_rows]
        axs[1].plot(rounds, train_accs, label="Treino Final", color="#2980b9", linewidth=1.8, marker="o", markersize=3)
        axs[1].plot(rounds, val_accs, label="Melhor Validacao", color="#27ae60", linewidth=1.8, marker="s", markersize=3)
        axs[1].plot(rounds, test_accs, label="Teste (130 fixos)", color="#e74c3c", linewidth=2.0, marker="^", markersize=4)
        axs[1].fill_between(rounds, train_accs, val_accs, color="#f39c12", alpha=0.15, label="Gap Treino-Val")
        axs[1].fill_between(rounds, val_accs, test_accs, color="#e74c3c", alpha=0.15, label="Gap Val-Teste")

    axs[1].set_title("Funil de Generalizacao: Treino vs Validacao vs Teste", fontsize=11, fontweight="bold")
    axs[1].set_xlabel("Rodada de Treinamento")
    axs[1].set_ylabel("Acuracia (%)")
    axs[1].grid(True, linestyle="--", alpha=0.5)
    axs[1].legend(fontsize=8, loc="lower left")

    # 3. Custo Computacional (Tempo e Épocas) vs Acurácia de Teste
    durations = [r["duracao_s"] for r in rows]
    epochs = [r["epochs_exec"] for r in rows]

    scatter = axs[2].scatter(
        durations,
        test_accs,
        c=[r.get(primary_param, 0) for r in rows],
        cmap="viridis",
        s=[max(45, ep * 3.2) for ep in epochs],
        alpha=0.85,
        edgecolors="black",
        linewidths=0.6,
    )
    axs[2].annotate(
        f"Top #1 (#{best_test['rodada']}: {_get_test_acc(best_test):.1f}%, {best_test['epochs_exec']} ep)",
        xy=(best_test["duracao_s"], _get_test_acc(best_test)),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=8,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="#fcf3cf", ec="#f1c40f", alpha=0.9),
    )
    axs[2].set_title(f"Custo Computacional vs Acc Teste (Bolha = Epocas)", fontsize=11, fontweight="bold")
    axs[2].set_xlabel("Duracao do Treino (segundos)")
    axs[2].set_ylabel("Acuracia no Teste (%)")
    axs[2].grid(True, linestyle="--", alpha=0.5)
    fig.colorbar(scatter, ax=axs[2], label=p_name)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
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

    path_insights = os.path.join(target_dir, "grid_search_insights.png")
    generated["insights"] = plot_global_insights(rows, path_insights)

    path_hyper = os.path.join(target_dir, "grid_search_hyperparams.png")
    res_hyper = plot_hyperparam_sensitivity(rows, path_hyper)
    if res_hyper:
        generated["hyperparams"] = res_hyper

    path_heatmaps = os.path.join(target_dir, "grid_search_heatmaps.png")
    res_heat = plot_heatmaps(rows, path_heatmaps)
    if res_heat:
        generated["heatmaps"] = res_heat

    return generated


def plot_grid_search_insights(rows: list[dict], output_path: str) -> str:
    """Funcao de compatibilidade que redireciona para plot_global_insights."""
    return plot_global_insights(rows, output_path)
