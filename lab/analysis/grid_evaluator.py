import csv
import os
import sys

from lab.analysis.grid_plots import plot_all_grid_visualizations


# ==============================================================================
#  Helpers de Formatação, Barras Visuais e Estatística
# ==============================================================================

PARAM_SPECS = [
    ("hidden", "Topologia (Neuronios Ocultos H)", "Neuronios (H)", lambda v: f"H={int(v)}"),
    ("val_ratio", "Proporcao de Validacao (Val Ratio)", "Val Ratio", lambda v: f"{v * 100:.1f}% ({v:.2f})"),
    ("lr", "Taxa de Aprendizado (Learn Rate)", "Taxa (η)", lambda v: f"η = {v}"),
    ("momentum", "Fator de Momentum", "Momentum (α)", lambda v: f"α = {v}"),
    ("patience", "Paciencia de Early Stopping", "Paciencia", lambda v: f"P = {int(v)}"),
    ("noise_rate", "Taxa de Ruido (Data Augmentation)", "Ruido", lambda v: f"{v * 100:.1f}%" if v > 0 else "0.0% (Limpo)"),
]


def _make_bar(val: float, low: float = 50.0, high: float = 100.0, width: int = 10) -> str:
    """Gera barra visual Unicode proporcional no intervalo [low, high]."""
    if high <= low:
        filled = width
    else:
        ratio = max(0.0, min(1.0, (val - low) / (high - low)))
        filled = int(round(ratio * width))
    return "█" * filled + "░" * (width - filled)


def _calc_group_stats(group: list[dict]) -> dict:
    """Calcula estatísticas completas de Validação, Teste e Custo Computacional para um grupo."""
    n = len(group)
    mean_val_acc = sum(r["best_val_acc"] for r in group) / n
    max_val_acc = max(r["best_val_acc"] for r in group)
    min_val_mse = min(r["best_val_mse"] for r in group)

    mean_test_acc = sum(r["test_acc"] for r in group) / n
    max_test_acc = max(r["test_acc"] for r in group)
    min_test_mse = min(r["test_mse"] for r in group)

    mean_gap_tv = sum(r["overfit_gap"] for r in group) / n
    mean_gap_vt = sum(r["val_test_gap"] for r in group) / n

    mean_epochs = sum(r["epochs_exec"] for r in group) / n
    mean_time = sum(r["duracao_s"] for r in group) / n
    sec_per_ep = mean_time / mean_epochs if mean_epochs > 0 else 0.0

    return {
        "n": n,
        "mean_val_acc": mean_val_acc,
        "max_val_acc": max_val_acc,
        "min_val_mse": min_val_mse,
        "mean_test_acc": mean_test_acc,
        "max_test_acc": max_test_acc,
        "min_test_mse": min_test_mse,
        "mean_gap_tv": mean_gap_tv,
        "mean_gap_vt": mean_gap_vt,
        "mean_epochs": mean_epochs,
        "mean_time": mean_time,
        "sec_per_ep": sec_per_ep,
    }


# ==============================================================================
#  Avaliador Principal de Grid Search
# ==============================================================================

class GridSearchEvaluator:
    """Orquestrador de relatorios analiticos visuais e graficos de Grid Search."""

    def __init__(self, target_dir: str):
        if not os.path.exists(target_dir):
            raise FileNotFoundError(f"Diretorio de Grid Search nao encontrado: {target_dir}")

        self.target_dir = target_dir.rstrip("/\\")
        self.summary_csv = os.path.join(self.target_dir, "summary.csv")
        if not os.path.exists(self.summary_csv):
            raise FileNotFoundError(f"Arquivo summary.csv nao encontrado em: {self.target_dir}")

        self.rows = self._load_rows()

    def _load_rows(self) -> list[dict]:
        """Lê o summary.csv e normaliza métricas de treino, validação e teste."""
        rows = []
        with open(self.summary_csv, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                train_acc = float(r["final_train_acc"]) if r.get("final_train_acc") else 0.0
                val_acc = float(r["best_val_acc"])
                val_mse = float(r["best_val_mse"])
                test_acc = float(r["test_acc"]) if r.get("test_acc") else val_acc
                test_mse = float(r["test_mse"]) if r.get("test_mse") else val_mse
                epochs = int(r["epochs_exec"])
                dur = float(r["duracao_s"])

                rows.append({
                    "rodada": int(r["rodada"]),
                    "hidden": int(r["hidden_neurons"]),
                    "lr": float(r["learn_rate"]),
                    "momentum": float(r["momentum"]),
                    "patience": int(r["patience"]),
                    "noise_rate": float(r["noise_rate"]) if r.get("noise_rate") else 0.0,
                    "val_ratio": float(r["val_ratio"]),
                    "epochs_exec": epochs,
                    "stopped_early": r["stopped_early"].lower() == "true",
                    "final_train_acc": train_acc,
                    "final_val_acc": float(r["final_val_acc"]) if r.get("final_val_acc") else val_acc,
                    "best_val_acc": val_acc,
                    "best_val_acc_epoch": int(r["best_val_acc_epoch"]) if r.get("best_val_acc_epoch") else 0,
                    "final_val_mse": float(r["final_val_mse"]) if r.get("final_val_mse") else 0.0,
                    "best_val_mse": val_mse,
                    "test_acc": test_acc,
                    "test_mse": test_mse,
                    "duracao_s": dur,
                    "sec_per_ep": round(dur / epochs, 3) if epochs > 0 else 0.0,
                    "checkpoint": r.get("checkpoint", ""),
                    "overfit_gap": round(train_acc - val_acc, 2),
                    "val_test_gap": round(val_acc - test_acc, 2),
                    "train_test_gap": round(train_acc - test_acc, 2),
                })
        return rows

    # --------------------------------------------------------------------------
    #  Geradores de Seção Markdown (Estilo Dashboard Visual)
    # --------------------------------------------------------------------------

    def _build_champions_section(self) -> str:
        """Gera comparativo visual dos 3 modelos de destaque da execução."""
        best_test = max(self.rows, key=lambda r: (r["test_acc"], -r["best_val_mse"], r["best_val_acc"]))
        best_val = min(self.rows, key=lambda r: (r["best_val_mse"], -r["best_val_acc"], -r["test_acc"]))

        max_t_acc = best_test["test_acc"]
        competitive = [r for r in self.rows if r["test_acc"] >= max_t_acc - 2.0]
        fastest_top = min(competitive, key=lambda r: (r["duracao_s"], r["epochs_exec"]))

        def _cfg(r: dict) -> str:
            return f"`H={r['hidden']}, LR={r['lr']}, M={r['momentum']}, Val={r['val_ratio']}, P={r['patience']}, Noise={r['noise_rate']}`"

        lines = [
            "## 1. Destaques da Execucao (Campeoes por Criterio)\n",
            "| Criterio | Rodada | Configuracao | Acc Teste (130 fixos) | MSE Teste | Acc Val (Melhor Ep) | MSE Val | Epocas / Tempo | Checkpoint |",
            "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
            (
                f"| **Campeao no Teste (Maior Acc)** | #{best_test['rodada']} | {_cfg(best_test)} | "
                f"**{best_test['test_acc']:.2f}%** | **{best_test['test_mse']:.5f}** | "
                f"{best_test['best_val_acc']:.2f}% (Ep {best_test['best_val_acc_epoch']}) | {best_test['best_val_mse']:.5f} | "
                f"{best_test['epochs_exec']} ep ({best_test['duracao_s']:.1f}s) | `{best_test['checkpoint'] or '-'}` |"
            ),
            (
                f"| **Campeao na Validacao (Menor MSE)** | #{best_val['rodada']} | {_cfg(best_val)} | "
                f"{best_val['test_acc']:.2f}% | {best_val['test_mse']:.5f} | "
                f"**{best_val['best_val_acc']:.2f}%** (Ep {best_val['best_val_acc_epoch']}) | **{best_val['best_val_mse']:.5f}** | "
                f"{best_val['epochs_exec']} ep ({best_val['duracao_s']:.1f}s) | `{best_val['checkpoint'] or '-'}` |"
            ),
            (
                f"| **Melhor Custo-Beneficio (Mais Rapido no Topo)** | #{fastest_top['rodada']} | {_cfg(fastest_top)} | "
                f"**{fastest_top['test_acc']:.2f}%** | {fastest_top['test_mse']:.5f} | "
                f"{fastest_top['best_val_acc']:.2f}% (Ep {fastest_top['best_val_acc_epoch']}) | {fastest_top['best_val_mse']:.5f} | "
                f"**{fastest_top['epochs_exec']} ep ({fastest_top['duracao_s']:.1f}s)** | `{fastest_top['checkpoint'] or '-'}` |"
            ),
        ]

        if os.path.exists(os.path.join(self.target_dir, "grid_search_insights.png")):
            lines.append("\n![Visao Global da Grade](./grid_search_insights.png)")

        return "\n".join(lines)

    def _build_param_tables(self, start_idx: int) -> tuple[str, int]:
        """Gera tabelas comparativas visuais (com barras Unicode + Validação + Teste + Custo)."""
        sections = []
        curr_idx = start_idx

        all_test_accs = [r["test_acc"] for r in self.rows]
        bar_low = max(40.0, min(all_test_accs) - 5.0) if all_test_accs else 50.0
        bar_high = min(100.0, max(all_test_accs) + 2.0) if all_test_accs else 90.0

        for key, title, header, fmt in PARAM_SPECS:
            unique_vals = sorted(list(set(r[key] for r in self.rows)))
            if len(unique_vals) <= 1:
                continue

            lines = [
                f"## {curr_idx}. Sensibilidade por {title}\n",
                (
                    f"| {header} | N | Acc Teste Media (Visual) | Acc Teste Max | MSE Teste Min | "
                    "Acc Val Media (Max) | MSE Val Min | Gap Val→Teste | Epocas Medias | Tempo Medio (s/ep) |"
                ),
                "| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]

            for val in unique_vals:
                group = [r for r in self.rows if r[key] == val]
                st = _calc_group_stats(group)
                bar = _make_bar(st["mean_test_acc"], low=bar_low, high=bar_high, width=10)
                lines.append(
                    f"| **{fmt(val)}** | {st['n']} | `{bar}` **{st['mean_test_acc']:.2f}%** | "
                    f"**{st['max_test_acc']:.2f}%** | {st['min_test_mse']:.5f} | "
                    f"{st['mean_val_acc']:.2f}% ({st['max_val_acc']:.2f}%) | {st['min_val_mse']:.5f} | "
                    f"{st['mean_gap_vt']:+.2f}% | {st['mean_epochs']:.1f} ep | "
                    f"{st['mean_time']:.2f}s (`{st['sec_per_ep']:.2f}s/ep`) |"
                )

            sections.append("\n".join(lines))
            curr_idx += 1

        if sections:
            visual_embeds = []
            if os.path.exists(os.path.join(self.target_dir, "grid_search_hyperparams.png")):
                visual_embeds.append("![Sensibilidade por Hiperparametro](./grid_search_hyperparams.png)")
            if os.path.exists(os.path.join(self.target_dir, "grid_search_heatmaps.png")):
                visual_embeds.append("![Mapas de Calor Bivariados](./grid_search_heatmaps.png)")
            if visual_embeds:
                sections.append("\n\n".join(visual_embeds))

        return "\n\n".join(sections), curr_idx

    def _build_funnel_and_overfitting_section(self, section_idx: int) -> str:
        """Gera funil visual de generalização (Treino -> Validação -> Teste) e distribuição de gaps."""
        n = len(self.rows)
        mean_tr = sum(r["final_train_acc"] for r in self.rows) / n
        mean_val = sum(r["best_val_acc"] for r in self.rows) / n
        mean_te = sum(r["test_acc"] for r in self.rows) / n

        max_tr = max(r["final_train_acc"] for r in self.rows)
        max_val = max(r["best_val_acc"] for r in self.rows)
        max_te = max(r["test_acc"] for r in self.rows)

        gap_tv = mean_tr - mean_val
        gap_vt = mean_val - mean_te
        gap_tt = mean_tr - mean_te

        bar_tr = _make_bar(mean_tr, low=50.0, high=100.0, width=20)
        bar_val = _make_bar(mean_val, low=50.0, high=100.0, width=20)
        bar_te = _make_bar(mean_te, low=50.0, high=100.0, width=20)

        best_gen_test = min(self.rows, key=lambda x: abs(x["val_test_gap"]))
        best_gen_train = min(self.rows, key=lambda x: abs(x["overfit_gap"]))

        gaps_tv = [r["overfit_gap"] for r in self.rows]
        c_low = sum(1 for g in gaps_tv if g <= 5.0)
        c_mid = sum(1 for g in gaps_tv if 5.0 < g <= 15.0)
        c_high = sum(1 for g in gaps_tv if g > 15.0)

        return (
            f"## {section_idx}. Funil de Generalizacao (Treino → Validacao → Teste)\n\n"
            "```text\n"
            f"  Treino Final      : [{bar_tr}] Media: {mean_tr:6.2f}% | Max: {max_tr:6.2f}%\n"
            f"                           │  (Queda media Treino → Validacao : -{gap_tv:.2f} pp)\n"
            f"                           ▼\n"
            f"  Melhor Validacao  : [{bar_val}] Media: {mean_val:6.2f}% | Max: {max_val:6.2f}%\n"
            f"                           │  (Queda media Validacao → Teste  : -{gap_vt:.2f} pp)\n"
            f"                           ▼\n"
            f"  Teste (130 fixos) : [{bar_te}] Media: {mean_te:6.2f}% | Max: {max_te:6.2f}%\n"
            "```\n\n"
            f"- **Gap Total Medio (Treino → Teste):** `{gap_tt:+.2f}%` (Treino→Val: `{gap_tv:+.2f}%` | Val→Teste: `{gap_vt:+.2f}%`)\n"
            f"- **Menor Desvio Val→Teste:** Rodada #{best_gen_test['rodada']} "
            f"(Val: `{best_gen_test['best_val_acc']:.2f}%` → Teste: `{best_gen_test['test_acc']:.2f}%` | Gap: `{best_gen_test['val_test_gap']:+.2f}%`)\n"
            f"- **Menor Sobreajuste Treino→Val:** Rodada #{best_gen_train['rodada']} "
            f"(Treino: `{best_gen_train['final_train_acc']:.2f}%` → Val: `{best_gen_train['best_val_acc']:.2f}%` | Gap: `{best_gen_train['overfit_gap']:+.2f}%`)\n"
            f"- **Distribuicao do Sobreajuste (Treino - Validacao):**\n"
            f"  - Baixo (`<= 5%`): **{c_low}/{n}** modelos\n"
            f"  - Moderado (`5% a 15%`): **{c_mid}/{n}** modelos\n"
            f"  - Alto (`> 15%`): **{c_high}/{n}** modelos\n"
        )

    def _build_top_k_table(self, sorted_rows: list[dict], section_idx: int, top_k: int = 10) -> str:
        """Gera a tabela com o ranking dos melhores modelos no Teste."""
        k = min(top_k, len(sorted_rows))
        all_test = [r["test_acc"] for r in self.rows]
        low_b = max(40.0, min(all_test) - 5.0) if all_test else 50.0
        high_b = min(100.0, max(all_test) + 2.0) if all_test else 90.0

        lines = [
            f"## {section_idx}. Ranking Top {k} Modelos (Ordenado por Acuracia de Teste)\n",
            (
                "| Pos | Rodada | H | Val | LR | Mom | P | Noise | Acc Teste (Visual) | MSE Teste | "
                "Acc Val (Ep) | MSE Val | Acc Treino | Epocas | Tempo (s/ep) | Checkpoint |"
            ),
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]

        for rank, r in enumerate(sorted_rows[:k], start=1):
            bar = _make_bar(r["test_acc"], low=low_b, high=high_b, width=8)
            lines.append(
                f"| **#{rank}** | #{r['rodada']} | {r['hidden']} | {r['val_ratio']:.2f} | {r['lr']} | {r['momentum']} | "
                f"{r['patience']} | {r['noise_rate']:.2f} | `{bar}` **{r['test_acc']:.2f}%** | {r['test_mse']:.5f} | "
                f"{r['best_val_acc']:.2f}% (Ep {r['best_val_acc_epoch']}) | {r['best_val_mse']:.5f} | "
                f"{r['final_train_acc']:.2f}% | {r['epochs_exec']} | {r['duracao_s']:.1f}s (`{r['sec_per_ep']:.2f}s`) | `{r['checkpoint'] or '-'}` |"
            )

        return "\n".join(lines)

    # --------------------------------------------------------------------------
    #  API Pública
    # --------------------------------------------------------------------------

    def generate_plots(self, output_png: str | None = None) -> dict[str, str]:
        """Gera todos os gráficos analíticos dedicados (insights, sensibilidade e heatmaps)."""
        return plot_all_grid_visualizations(self.rows, self.target_dir)

    def generate_report(self, output_md: str | None = None) -> str:
        """Monta o relatório ANALISE_GRID_SEARCH.md combinando as seções visuais."""
        dest_md = output_md or os.path.join(self.target_dir, "ANALISE_GRID_SEARCH.md")
        sorted_rows = sorted(
            self.rows,
            key=lambda x: (-x["test_acc"], x["test_mse"], x["best_val_mse"], -x["best_val_acc"]),
        )

        total_time = sum(r["duracao_s"] for r in self.rows)
        total_epochs = sum(r["epochs_exec"] for r in self.rows)

        header = (
            "# Relatorio Analitico e Visual — Grid Search\n\n"
            f"- **Total de Modelos Treinados:** `{len(self.rows)}` combinacoes\n"
            f"- **Esforco Computacional Total:** `{total_epochs}` epocas acumuladas em `{total_time:.1f}s` (`{total_time / 60:.1f} min`)\n"
            "- **Conjunto de Avaliacao Final:** `130 amostras fixas de Teste` (ultimos 5 alfabetos completos, indices `1196..1325`)\n"
        )
        champions_sec = self._build_champions_section()
        param_sec, next_idx = self._build_param_tables(start_idx=2)
        funnel_sec = self._build_funnel_and_overfitting_section(section_idx=next_idx)
        top_k_sec = self._build_top_k_table(sorted_rows, section_idx=next_idx + 1, top_k=10)

        sections = [header, champions_sec]
        if param_sec:
            sections.append(param_sec)
        sections.extend([funnel_sec, top_k_sec])

        with open(dest_md, "w", encoding="utf-8") as f:
            f.write("\n\n".join(sections) + "\n")

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

# ==============================================================================
# Integrantes do Grupo (EACH-USP | Inteligência Artificial 2026):
# - Guilherme Oliveira de Souza (Nº USP: 14671388)
# - Paulo Rogerio Andrade de Oliveira (Nº USP: 9365783)
# - Victor Hugo Antunes Valadares (Nº USP: 14760317)
# ==============================================================================
