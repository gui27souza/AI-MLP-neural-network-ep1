import csv
import os
import sys
from collections import defaultdict

from lab.analysis.grid_plots import plot_all_grid_visualizations


# ==============================================================================
#  Helpers de Formatação e Estatística
# ==============================================================================

PARAM_SPECS = [
    ("hidden", "Topologia (Neuronios Ocultos H)", "Neuronios (H)", lambda v: f"H={int(v)}"),
    ("val_ratio", "Proporcao de Validacao (Val Ratio)", "Val Ratio", lambda v: f"{v * 100:.1f}% ({v:.2f})"),
    ("lr", "Taxa de Aprendizado (Learn Rate)", "Taxa (η)", lambda v: f"η = {v}"),
    ("momentum", "Fator de Momentum", "Momentum (α)", lambda v: f"α = {v}"),
    ("patience", "Paciencia de Early Stopping", "Paciencia", lambda v: f"P = {int(v)}"),
    ("noise_rate", "Taxa de Ruido (Data Augmentation)", "Ruido", lambda v: f"{v * 100:.1f}%" if v > 0 else "0.0% (Limpo)"),
]


def _calc_stats(group: list[dict]) -> tuple[float, float, float, float, float]:
    """Retorna (acc_media, acc_max, gap_medio, mse_min, tempo_medio) de um grupo."""
    n = len(group)
    mean_acc = sum(r["best_val_acc"] for r in group) / n
    max_acc = max(r["best_val_acc"] for r in group)
    mean_gap = sum(r["overfit_gap"] for r in group) / n
    min_mse = min(r["best_val_mse"] for r in group)
    mean_time = sum(r["duracao_s"] for r in group) / n
    return mean_acc, max_acc, mean_gap, min_mse, mean_time


# ==============================================================================
#  Avaliador Principal de Grid Search
# ==============================================================================

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
        """Lê o summary.csv e normaliza métricas e gap de overfitting."""
        rows = []
        with open(self.summary_csv, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                train_acc = float(r["final_train_acc"]) if r.get("final_train_acc") else 0.0
                val_acc = float(r["best_val_acc"])
                rows.append({
                    "rodada": int(r["rodada"]),
                    "hidden": int(r["hidden_neurons"]),
                    "lr": float(r["learn_rate"]),
                    "momentum": float(r["momentum"]),
                    "patience": int(r["patience"]),
                    "noise_rate": float(r["noise_rate"]) if r.get("noise_rate") else 0.0,
                    "val_ratio": float(r["val_ratio"]),
                    "epochs_exec": int(r["epochs_exec"]),
                    "stopped_early": r["stopped_early"].lower() == "true",
                    "final_train_acc": train_acc,
                    "final_val_acc": float(r["final_val_acc"]) if r.get("final_val_acc") else val_acc,
                    "best_val_acc": val_acc,
                    "best_val_acc_epoch": int(r["best_val_acc_epoch"]) if r.get("best_val_acc_epoch") else 0,
                    "final_val_mse": float(r["final_val_mse"]) if r.get("final_val_mse") else 0.0,
                    "best_val_mse": float(r["best_val_mse"]),
                    "duracao_s": float(r["duracao_s"]),
                    "checkpoint": r.get("checkpoint", ""),
                    "overfit_gap": round(train_acc - val_acc, 2),
                })
        return rows

    # --------------------------------------------------------------------------
    #  Geradores de Seção Markdown (Modulares e de baixa tensão cognitiva)
    # --------------------------------------------------------------------------

    @staticmethod
    def _build_champion_section(best: dict) -> str:
        """Gera a seção do modelo campeão absoluto."""
        early_str = "Sim" if best["stopped_early"] else "Nao"
        ckpt_str = best["checkpoint"] or "Descartado por limiar de acuracia"
        return (
            "## 1. Melhor Modelo da Execucao (Top 1)\n\n"
            f"- **Rodada:** #{best['rodada']}\n"
            f"- **Topologia:** `H = {best['hidden']}`\n"
            f"- **Hiperparametros:** `Val Ratio = {best['val_ratio']}`, `LR = {best['lr']}`, "
            f"`Momentum = {best['momentum']}`, `Patience = {best['patience']}`, `Noise = {best['noise_rate']}`\n"
            f"- **Melhor Acuracia de Validacao:** **{best['best_val_acc']:.2f}%** (alcancada na epoca {best['best_val_acc_epoch']})\n"
            f"- **Acuracia de Treino Final:** {best['final_train_acc']:.2f}% (Gap Treino-Val: **{best['overfit_gap']:+.2f}%**)\n"
            f"- **Menor MSE de Validacao:** **{best['best_val_mse']:.5f}**\n"
            f"- **Epocas Executadas:** {best['epochs_exec']} ({best['duracao_s']:.2f}s) | Parada Antecipada: `{early_str}`\n"
            f"- **Checkpoint Salvo:** `{ckpt_str}`\n"
        )

    def _build_param_tables(self, start_idx: int) -> tuple[str, int]:
        """Gera tabelas comparativas para todos os hiperparâmetros que variaram."""
        sections = []
        curr_idx = start_idx

        for key, title, header, fmt in PARAM_SPECS:
            unique_vals = sorted(list(set(r[key] for r in self.rows)))
            if len(unique_vals) <= 1:
                continue

            lines = [
                f"## {curr_idx}. Comparativo por {title}\n",
                f"| {header} | Total Modelos | Acuracia Media | Acuracia Maxima | Gap Overfit Medio | MSE Minimo | Tempo Medio (s) |",
                "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]

            for val in unique_vals:
                group = [r for r in self.rows if r[key] == val]
                mean_acc, max_acc, mean_gap, min_mse, mean_time = _calc_stats(group)
                lines.append(
                    f"| **{fmt(val)}** | {len(group)} | {mean_acc:.2f}% | **{max_acc:.2f}%** | "
                    f"{mean_gap:+.2f}% | **{min_mse:.5f}** | {mean_time:.2f}s |"
                )

            sections.append("\n".join(lines))
            curr_idx += 1

        return "\n\n".join(sections), curr_idx

    def _build_overfitting_section(self, section_idx: int) -> str:
        """Gera diagnóstico de gap de generalização entre treino e validação."""
        gaps = [r["overfit_gap"] for r in self.rows]
        mean_gap = sum(gaps) / len(gaps)
        best_gen = min(self.rows, key=lambda x: abs(x["overfit_gap"]))

        c_low = sum(1 for g in gaps if g <= 5.0)
        c_mid = sum(1 for g in gaps if 5.0 < g <= 15.0)
        c_high = sum(1 for g in gaps if g > 15.0)
        total = len(self.rows)

        return (
            f"## {section_idx}. Diagnostico de Generalizacao & Overfitting\n\n"
            f"- **Gap Medio Global (Treino - Validacao):** **{mean_gap:+.2f}%**\n"
            f"- **Modelo Mais Generalizador (Menor Gap):** Rodada #{best_gen['rodada']} "
            f"(Treino: {best_gen['final_train_acc']:.2f}% | Val: {best_gen['best_val_acc']:.2f}% | Gap: **{best_gen['overfit_gap']:+.2f}%**)\n"
            "- **Distribuicao da Capacidade de Generalizacao:**\n"
            f"  - Generalizacao Alta (Gap <= 5%): **{c_low}/{total}** modelos\n"
            f"  - Sobreajuste Moderado (5% < Gap <= 15%): **{c_mid}/{total}** modelos\n"
            f"  - Sobreajuste Acentuado (Gap > 15%): **{c_high}/{total}** modelos\n"
        )

    @staticmethod
    def _build_top_k_table(sorted_rows: list[dict], section_idx: int, top_k: int = 10) -> str:
        """Gera a tabela com o ranking dos melhores modelos."""
        k = min(top_k, len(sorted_rows))
        lines = [
            f"## {section_idx}. Top {k} Melhores Modelos da Execucao\n",
            "| Pos | Rodada | H | Ratio | LR | Momentum | P | Noise | Acc Treino | Acc Val | Gap Overfit | MSE Val | Epocas | Tempo | Checkpoint |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]

        for rank, r in enumerate(sorted_rows[:k], start=1):
            lines.append(
                f"| #{rank} | #{r['rodada']} | {r['hidden']} | {r['val_ratio']:.2f} | {r['lr']} | {r['momentum']} | "
                f"{r['patience']} | {r['noise_rate']:.2f} | {r['final_train_acc']:.2f}% | **{r['best_val_acc']:.2f}%** | "
                f"{r['overfit_gap']:+.2f}% | {r['best_val_mse']:.5f} | {r['epochs_exec']} | {r['duracao_s']:.1f}s | `{r['checkpoint']}` |"
            )

        return "\n".join(lines)

    @staticmethod
    def _build_artifacts_section(section_idx: int) -> str:
        """Lista os artefatos visuais gerados."""
        return (
            f"## {section_idx}. Paineis Visuais Gerados\n\n"
            "- **Visao Global (Fronteira MSE, Overfitting e Custo):** `grid_search_insights.png`\n"
            "- **Sensibilidade Parametrica Individual (Acc vs Hiperparametros):** `grid_search_hyperparams.png`\n"
            "- **Mapas de Calor Bivariados (Interacao de Pares):** `grid_search_heatmaps.png` (se >= 2 variaveis)\n"
        )

    # --------------------------------------------------------------------------
    #  API Pública
    # --------------------------------------------------------------------------

    def generate_plots(self, output_png: str | None = None) -> dict[str, str]:
        """Gera todos os gráficos analíticos dedicados (insights, sensibilidade e heatmaps)."""
        return plot_all_grid_visualizations(self.rows, self.target_dir)

    def generate_report(self, output_md: str | None = None) -> str:
        """Monta o relatório ANALISE_GRID_SEARCH.md combinando as seções."""
        dest_md = output_md or os.path.join(self.target_dir, "ANALISE_GRID_SEARCH.md")
        sorted_rows = sorted(self.rows, key=lambda x: (-x["best_val_acc"], x["best_val_mse"]))

        header = f"# Relatorio Analitico - Grid Search\n\nTotal de Modelos Treinados: {len(self.rows)}\n"
        champion_sec = self._build_champion_section(sorted_rows[0])
        param_sec, next_idx = self._build_param_tables(start_idx=2)
        overfit_sec = self._build_overfitting_section(section_idx=next_idx)
        top_k_sec = self._build_top_k_table(sorted_rows, section_idx=next_idx + 1, top_k=10)
        artifacts_sec = self._build_artifacts_section(section_idx=next_idx + 2)

        sections = [header, champion_sec]
        if param_sec:
            sections.append(param_sec)
        sections.extend([overfit_sec, top_k_sec, artifacts_sec])

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
