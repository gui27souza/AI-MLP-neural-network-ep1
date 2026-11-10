import csv
import json
import os
import shutil
import sys
from collections import defaultdict

import numpy as np

from src.dataset_loader import DatasetLoader
from src.funcs import sigmoide_bipolar, deriv_sigmoide_bipolar
from src.mlp_persister import MLPPersister
from src.models import MLP


class MLPEvaluator:
    """Responsável por avaliar modelos persistidos, gerar métricas detalhadas e gráficos de treinamento."""

    @staticmethod
    def _resolve_unique_dir(dir_path: str) -> str:
        """Se o diretório já existir e contiver arquivos, adiciona sufixo incremental (_v2, _v3, ...) para não sobrescrever."""
        if not os.path.exists(dir_path):
            return dir_path

        if not os.listdir(dir_path):
            return dir_path

        version = 2
        while os.path.exists(f"{dir_path}_v{version}"):
            if not os.listdir(f"{dir_path}_v{version}"):
                return f"{dir_path}_v{version}"
            version += 1
        return f"{dir_path}_v{version}"

    def _resolve_output_dir(self, output_dir: str | None = None) -> str:
        """Define o diretório final de resultados, garantindo blindagem contra sobrescrita."""
        if output_dir is not None:
            return self._resolve_unique_dir(output_dir)

        # Se o modelo avaliado já está dentro de uma pasta em results/, mantém nela mesma
        expected_dir = os.path.abspath(os.path.join("results", self.model_name))
        current_dir = os.path.abspath(self.model_dir)
        if current_dir == expected_dir or current_dir.startswith(expected_dir + os.sep):
            return self.model_dir

        return self._resolve_unique_dir(os.path.join("results", self.model_name))

    def __init__(self, model_path: str, output_dir: str | None = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Arquivo de modelo não encontrado: {model_path}")

        self.model_path = model_path
        self.model_dir = os.path.dirname(os.path.abspath(model_path))
        self.model_filename = os.path.basename(model_path)
        self.model_name = os.path.splitext(self.model_filename)[0]

        self.output_dir = self._resolve_output_dir(output_dir)

        with open(model_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

        self.metrics = self.raw_data.get("metrics", {})
        self.history = self.metrics.get("history", {})

        self.mlp = MLPPersister.load(
            filepath=model_path,
            activation_func=sigmoide_bipolar,
            derivative_activation=deriv_sigmoide_bipolar,
        )

    def get_val_ratio(self) -> float:
        """Determina a proporção de validação original usada no treino."""
        ds_info = self.metrics.get("dataset_info", {})
        train_samples = ds_info.get("train_samples", 0)
        val_samples = ds_info.get("val_samples", 0)
        total = train_samples + val_samples
        if total > 0 and val_samples > 0:
            return round(val_samples / total, 2)
        return 0.3

    def evaluate_test_set(
        self, val_ratio: float | None = None
    ) -> dict:
        """Executa a inferência no conjunto de teste e calcula matriz de confusão e métricas por classe."""
        ratio = val_ratio if val_ratio is not None else self.get_val_ratio()

        _, _, test_x, test_y, class_labels = DatasetLoader.load_caracteres_completo(val_ratio=ratio)

        num_samples = len(test_x)
        num_classes = len(class_labels)

        # Matriz de confusão [real][predito]
        confusion_matrix = [[0] * num_classes for _ in range(num_classes)]

        class_total = defaultdict(int)
        class_correct = defaultdict(int)
        confusions = defaultdict(int)

        total_correct = 0
        total_squared_error = 0.0

        continuous_rows = []
        discrete_rows = []

        for sample_idx, (x, target) in enumerate(zip(test_x, test_y)):
            output = self.mlp.predict(x)

            actual_idx = max(range(len(target)), key=lambda i: target[i])
            pred_idx = max(range(len(output)), key=lambda i: output[i])

            actual_label = class_labels[actual_idx]
            pred_label = class_labels[pred_idx]

            confusion_matrix[actual_idx][pred_idx] += 1
            class_total[actual_label] += 1

            # Erro quadrático da amostra
            sample_mse = sum((t - o) ** 2 for t, o in zip(target, output)) / len(target)
            total_squared_error += sample_mse

            is_correct = (actual_idx == pred_idx)
            if is_correct:
                total_correct += 1
                class_correct[actual_label] += 1
            else:
                confusions[(actual_label, pred_label)] += 1

            continuous_rows.append({
                "sample_idx": sample_idx,
                "actual_label": actual_label,
                "pred_label": pred_label,
                "correct": int(is_correct),
                "output": [float(val) for val in output],
            })

            discrete_vec = [-1.0] * num_classes
            discrete_vec[pred_idx] = 1.0
            discrete_rows.append({
                "sample_idx": sample_idx,
                "actual_label": actual_label,
                "pred_label": pred_label,
                "correct": int(is_correct),
                "output": discrete_vec,
            })

        overall_acc = (total_correct / num_samples) * 100 if num_samples > 0 else 0.0
        overall_mse = total_squared_error / num_samples if num_samples > 0 else 0.0

        class_accuracies = {}
        for label in class_labels:
            tot = class_total[label]
            corr = class_correct[label]
            class_accuracies[label] = {
                "total": tot,
                "correct": corr,
                "accuracy": (corr / tot * 100) if tot > 0 else 0.0,
            }

        sorted_confusions = sorted(confusions.items(), key=lambda item: item[1], reverse=True)

        return {
            "num_samples": num_samples,
            "overall_accuracy": overall_acc,
            "overall_mse": overall_mse,
            "confusion_matrix": confusion_matrix,
            "class_labels": class_labels,
            "class_accuracies": class_accuracies,
            "top_confusions": sorted_confusions,
            "val_ratio": ratio,
            "continuous_predictions": continuous_rows,
            "discrete_predictions": discrete_rows,
        }

    def _build_report_text(self, eval_results: dict) -> str:
        """Gera o texto estruturado do relatório de avaliação."""
        class_acc = eval_results["class_accuracies"]
        confusions = eval_results["top_confusions"]

        lines = [
            "=" * 78,
            f"  RELATORIO DE AVALIACAO DO MODELO: {self.model_filename}",
            "=" * 78,
            f"  Amostras de Teste/Validacao : {eval_results['num_samples']} (Ratio={eval_results['val_ratio']})",
            f"  Acuracia Geral              : {eval_results['overall_accuracy']:.2f}%",
            f"  MSE de Teste                : {eval_results['overall_mse']:.5f}",
            "-" * 78,
            "",
            "[TOP 5 CARACTERES COM MAIOR DIFICULDADE (Menor Acuracia)]",
        ]

        sorted_by_acc = sorted(class_acc.items(), key=lambda x: (x[1]["accuracy"], x[0]))
        for label, data in sorted_by_acc[:5]:
            lines.append(f"  Letra '{label}': {data['accuracy']:>6.2f}% ({data['correct']}/{data['total']} acertos)")

        lines.extend([
            "",
            "[TOP 5 CARACTERES MAIS RECONHECIDOS (Maior Acuracia)]",
        ])
        for label, data in reversed(sorted_by_acc[-5:]):
            lines.append(f"  Letra '{label}': {data['accuracy']:>6.2f}% ({data['correct']}/{data['total']} acertos)")

        if confusions:
            lines.extend([
                "",
                "[MAIORES CONFUSOES OBSERVADAS (Real -> Predito)]",
            ])
            for (actual, pred), count in confusions[:10]:
                lines.append(f"  Real '{actual}' classificado como '{pred}': {count} ocorrencia(s)")
        else:
            lines.extend([
                "",
                "Nenhuma confusao observada (100% de acertos).",
            ])

        lines.append("=" * 78)
        return "\n".join(lines)

    def print_evaluation_report(self, eval_results: dict) -> None:
        """Exibe no terminal o relatório detalhado de acerto e confusão do modelo."""
        print("\n" + self._build_report_text(eval_results))

    def save_evaluation_report(self, eval_results: dict, output_dir: str | None = None) -> str:
        """Salva o resumo textual da avaliação no diretório de resultados."""
        target_dir = output_dir if output_dir is not None else self.output_dir
        os.makedirs(target_dir, exist_ok=True)
        report_path = os.path.join(target_dir, "evaluation_report.txt")

        report_text = self._build_report_text(eval_results)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text + "\n")

        print(f"Relatorio textual salvo em: {report_path}")
        return report_path

    def copy_model_to_results(self, output_dir: str | None = None) -> str:
        """Copia o arquivo .json do modelo para centralizar todos os artefatos de entrega."""
        target_dir = output_dir if output_dir is not None else self.output_dir
        os.makedirs(target_dir, exist_ok=True)
        dest_path = os.path.join(target_dir, self.model_filename)

        if os.path.abspath(self.model_path) != os.path.abspath(dest_path):
            shutil.copy2(self.model_path, dest_path)
            print(f"Modelo .json copiado para a pasta de resultados: {dest_path}")

        return dest_path

    def plot_learning_curves(self, output_dir: str | None = None) -> str | None:
        """Gera o gráfico de curvas de aprendizado (MSE e Acurácia) se o histórico estiver presente."""
        train_mse = self.history.get("train_mse", [])
        val_mse = self.history.get("val_mse", [])
        train_acc = self.history.get("train_acc", [])
        val_acc = self.history.get("val_acc", [])

        if not train_mse:
            print(
                f"\nAviso: Historico de treinamento nao encontrado em {self.model_filename} "
                "(modelo salvo antes da ativacao do history)."
            )
            return None

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("\nAviso: matplotlib nao instalado. Instale com: pip install matplotlib")
            return None

        target_dir = output_dir if output_dir is not None else self.output_dir
        os.makedirs(target_dir, exist_ok=True)
        epochs = list(range(1, len(train_mse) + 1))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Curva de MSE
        ax1.plot(epochs, train_mse, label="MSE Treino", color="#1f77b4", linewidth=2)
        if val_mse:
            ax1.plot(epochs, val_mse, label="MSE Validacao", color="#ff7f0e", linewidth=2, linestyle="--")

        saved_epoch = self.metrics.get("stopping_info", {}).get("saved_epoch")
        if saved_epoch and saved_epoch <= len(epochs):
            ax1.axvline(x=saved_epoch, color="#2ca02c", linestyle=":", label=f"Melhor Epoca ({saved_epoch})")

        ax1.set_title("Curva de Aprendizado - MSE")
        ax1.set_xlabel("Epocas")
        ax1.set_ylabel("MSE")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend()

        # 2. Curva de Acurácia
        if train_acc:
            ax2.plot(epochs, train_acc, label="Acc Treino (%)", color="#1f77b4", linewidth=2)
        if val_acc:
            ax2.plot(epochs, val_acc, label="Acc Validacao (%)", color="#ff7f0e", linewidth=2, linestyle="--")

        if saved_epoch and saved_epoch <= len(epochs):
            ax2.axvline(x=saved_epoch, color="#2ca02c", linestyle=":", label=f"Melhor Epoca ({saved_epoch})")

        ax2.set_title("Curva de Acuracia (%)")
        ax2.set_xlabel("Epocas")
        ax2.set_ylabel("Acuracia (%)")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend()

        plt.suptitle(f"Evolucao do Treinamento - {self.model_filename}", fontsize=13)
        plt.tight_layout()

        output_path = os.path.join(target_dir, "learning_curves.png")
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"\nGrafico de curvas de aprendizado salvo em: {output_path}")
        return output_path

    def plot_confusion_matrix(self, eval_results: dict, output_dir: str | None = None) -> str | None:
        """Gera o mapa de calor da Matriz de Confusão 26 x 26."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("\nAviso: matplotlib nao instalado. Instale com: pip install matplotlib")
            return None

        target_dir = output_dir if output_dir is not None else self.output_dir
        os.makedirs(target_dir, exist_ok=True)

        matrix = np.array(eval_results["confusion_matrix"])
        labels = eval_results["class_labels"]

        fig, ax = plt.subplots(figsize=(12, 10))
        cax = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
        fig.colorbar(cax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)

        ax.set_xlabel("Classe Predita", fontsize=11, fontweight="bold")
        ax.set_ylabel("Classe Real", fontsize=11, fontweight="bold")
        ax.set_title(
            f"Matriz de Confusao - {self.model_filename}\nAcuracia Geral: {eval_results['overall_accuracy']:.2f}%",
            fontsize=13,
            pad=15,
        )

        # Adiciona os números em cada célula quando > 0
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = matrix[i, j]
                if val > 0:
                    color = "white" if val > matrix.max() / 2 else "black"
                    ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8)

        plt.tight_layout()
        output_path = os.path.join(target_dir, "confusion_matrix.png")
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"Matriz de confusao salva em: {output_path}")
        return output_path

    def _export_hyperparameters(self, output_dir: str, eval_results: dict | None = None) -> str:
        """Gera o arquivo hiperparametros.txt com todos os parametros da rede e resultados."""
        filepath = os.path.join(output_dir, "hiperparametros.txt")

        hp = self.raw_data.get("hyperparameters", {})
        training_params = self.metrics.get("training_params", {})
        stopping_info = self.metrics.get("stopping_info", {})
        ds_info = self.metrics.get("dataset_info", {})

        learn_rate = hp.get("learn_rate", getattr(self.mlp, "learn_rate", "N/A"))
        momentum = hp.get("momentum", getattr(self.mlp, "momentum", "N/A"))
        num_inputs = hp.get("num_inputs", self.mlp.num_inputs)
        num_outputs = hp.get("num_outputs", self.mlp.num_outputs)
        hidden_neurons = hp.get("hidden_neurons", [layer.num_neurons for layer in self.mlp.hidden_layers])

        max_epochs = training_params.get("max_epochs", "N/A")
        target_mse = training_params.get("target_mse", "N/A")
        patience = training_params.get("patience", "N/A")
        epochs_executed = self.metrics.get("epochs_executed", len(self.history.get("train_mse", [])))
        duration = self.metrics.get("duration_seconds", "N/A")

        train_samples = ds_info.get("train_samples", "N/A")
        val_samples = ds_info.get("val_samples", "N/A")
        val_ratio = eval_results.get("val_ratio", self.get_val_ratio()) if eval_results else self.get_val_ratio()

        train_mse = self.metrics.get("train_mse", "N/A")
        train_acc = self.metrics.get("train_acc", "N/A")
        val_mse = self.metrics.get("val_mse", "N/A")
        val_acc = self.metrics.get("val_acc", "N/A")

        test_mse = eval_results.get("overall_mse", "N/A") if eval_results else "N/A"
        test_acc = eval_results.get("overall_accuracy", "N/A") if eval_results else "N/A"

        lines = [
            "=" * 78,
            "                    CONFIGURACAO DE HIPERPARAMETROS DO MODELO",
            "=" * 78,
            f"Modelo               : {self.model_filename}",
            f"Data de Geracao      : {self.raw_data.get('created_at', 'N/A')}",
            f"Tempo de Treinamento : {duration:.2f}s" if isinstance(duration, (int, float)) else f"Tempo de Treinamento : {duration}",
            "",
            "[1. ARQUITETURA DA REDE]",
            f"  - Entradas (n)              : {num_inputs} (caracteres 10x12 normalizados [-1, 1])",
            f"  - Camada(s) Oculta(s) (p)   : {hidden_neurons} neuronios",
            f"  - Saidas (m)                : {num_outputs} neuronios (classes A-Z)",
            "  - Funcao de Ativacao        : Sigmoide Bipolar (faixa [-1, 1])",
            "  - Derivada da Ativacao      : f'(x) = 0.5 * (1 + f(x)) * (1 - f(x))",
            "",
            "[2. HIPERPARAMETROS DE TREINAMENTO]",
            f"  - Taxa de Aprendizado (eta) : {learn_rate}",
            f"  - Termo de Momento (alpha)  : {momentum}",
            f"  - Maximo de Epocas          : {max_epochs}",
            f"  - Epocas Executadas         : {epochs_executed}",
            f"  - Criterio de Parada        : Early Stopping (Paciencia={patience}, Min Delta MSE={target_mse})",
            f"  - Parada Acionada           : {stopping_info.get('stopped_early', False)}",
            f"  - Motivo da Parada          : {stopping_info.get('stop_reason', 'N/A')}",
            f"  - Epoca do Melhor Modelo    : {stopping_info.get('saved_epoch', 'N/A')}",
            "",
            "[3. DADOS E AMOSTRAGEM]",
            "  - Dataset                   : caracteres-completo (EACH-USP)",
            f"  - Amostras de Treinamento   : {train_samples}",
            f"  - Amostras de Validacao     : {val_samples} (Ratio={val_ratio})",
            f"  - Amostras de Teste         : {eval_results.get('num_samples', 'N/A') if eval_results else 'N/A'}",
            "",
            "[4. DESEMPENHO ALCANCADO]",
            f"  - MSE de Treinamento        : {train_mse:.6f}" if isinstance(train_mse, (int, float)) else f"  - MSE de Treinamento        : {train_mse}",
            f"  - Acuracia de Treinamento   : {train_acc:.2f}%" if isinstance(train_acc, (int, float)) else f"  - Acuracia de Treinamento   : {train_acc}",
            f"  - MSE de Validacao          : {val_mse:.6f}" if isinstance(val_mse, (int, float)) else f"  - MSE de Validacao          : {val_mse}",
            f"  - Acuracia de Validacao     : {val_acc:.2f}%" if isinstance(val_acc, (int, float)) else f"  - Acuracia de Validacao     : {val_acc}",
            f"  - MSE de Teste              : {test_mse:.6f}" if isinstance(test_mse, (int, float)) else f"  - MSE de Teste              : {test_mse}",
            f"  - Acuracia de Teste         : {test_acc:.2f}%" if isinstance(test_acc, (int, float)) else f"  - Acuracia de Teste         : {test_acc}",
            "=" * 78,
        ]

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return filepath

    def _export_weights(self, output_dir: str) -> dict[str, str]:
        """Gera os arquivos de pesos no formato matricial de Fausett (V e W)."""
        saved_paths = {}

        # 1. Pesos Finais
        final_v_mats, final_w_mat = self.mlp.to_fausett_matrices()

        for idx, v_mat in enumerate(final_v_mats):
            v_name = "pesos_finais_V.csv" if len(final_v_mats) == 1 else f"pesos_finais_V{idx+1}.csv"
            v_path = os.path.join(output_dir, v_name)
            np.savetxt(v_path, v_mat, delimiter=",", fmt="%.8f")
            saved_paths[v_name] = v_path

        w_path = os.path.join(output_dir, "pesos_finais_W.csv")
        np.savetxt(w_path, final_w_mat, delimiter=",", fmt="%.8f")
        saved_paths["pesos_finais_W.csv"] = w_path

        combined_final_path = os.path.join(output_dir, "pesos_finais.csv")
        with open(combined_final_path, "w", encoding="utf-8") as f:
            for idx, v_mat in enumerate(final_v_mats):
                suffix = f" {idx+1}" if len(final_v_mats) > 1 else ""
                f.write(f"# MATRIZ V{suffix} (Camada Oculta: {v_mat.shape[0]} neuronios x {v_mat.shape[1]} colunas [col 0 = bias])\n")
                np.savetxt(f, v_mat, delimiter=",", fmt="%.8f")
                f.write("\n")
            f.write(f"# MATRIZ W (Camada de Saida: {final_w_mat.shape[0]} neuronios x {final_w_mat.shape[1]} colunas [col 0 = bias])\n")
            np.savetxt(f, final_w_mat, delimiter=",", fmt="%.8f")
        saved_paths["pesos_finais.csv"] = combined_final_path

        # 2. Pesos Iniciais
        if self.mlp.initial_weights is not None:
            init_v_mats, init_w_mat = self.mlp.to_fausett_matrices(self.mlp.initial_weights)

            for idx, v_mat in enumerate(init_v_mats):
                v_name = "pesos_iniciais_V.csv" if len(init_v_mats) == 1 else f"pesos_iniciais_V{idx+1}.csv"
                v_path = os.path.join(output_dir, v_name)
                np.savetxt(v_path, v_mat, delimiter=",", fmt="%.8f")
                saved_paths[v_name] = v_path

            w_init_path = os.path.join(output_dir, "pesos_iniciais_W.csv")
            np.savetxt(w_init_path, init_w_mat, delimiter=",", fmt="%.8f")
            saved_paths["pesos_iniciais_W.csv"] = w_init_path

            combined_init_path = os.path.join(output_dir, "pesos_iniciais.csv")
            with open(combined_init_path, "w", encoding="utf-8") as f:
                for idx, v_mat in enumerate(init_v_mats):
                    suffix = f" {idx+1}" if len(init_v_mats) > 1 else ""
                    f.write(f"# MATRIZ V{suffix} INICIAL (Camada Oculta: {v_mat.shape[0]} neuronios x {v_mat.shape[1]} colunas [col 0 = bias])\n")
                    np.savetxt(f, v_mat, delimiter=",", fmt="%.8f")
                    f.write("\n")
                f.write(f"# MATRIZ W INICIAL (Camada de Saida: {init_w_mat.shape[0]} neuronios x {init_w_mat.shape[1]} colunas [col 0 = bias])\n")
                np.savetxt(f, init_w_mat, delimiter=",", fmt="%.8f")
            saved_paths["pesos_iniciais.csv"] = combined_init_path
        else:
            dummy_init_path = os.path.join(output_dir, "pesos_iniciais.csv")
            with open(dummy_init_path, "w", encoding="utf-8") as f:
                f.write("# Pesos iniciais nao encontrados no arquivo .json (modelo gerado antes da atualizacao).\n")
                f.write("# Para modelos gerados a partir desta versao, os pesos iniciais sao persistidos automaticamente.\n")
            saved_paths["pesos_iniciais.csv"] = dummy_init_path

        return saved_paths

    def _export_history(self, output_dir: str) -> str:
        """Gera o historico_erro_treinamento.csv com MSE e acuracia por epoca."""
        filepath = os.path.join(output_dir, "historico_erro_treinamento.csv")
        train_mse = self.history.get("train_mse", [])
        val_mse = self.history.get("val_mse", [])
        train_acc = self.history.get("train_acc", [])
        val_acc = self.history.get("val_acc", [])

        with open(filepath, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoca", "mse_treinamento", "mse_validacao", "acuracia_treinamento", "acuracia_validacao"])

            num_epochs = max(len(train_mse), len(val_mse), len(train_acc), len(val_acc))
            for i in range(num_epochs):
                e = i + 1
                tmse = f"{train_mse[i]:.6f}" if i < len(train_mse) else ""
                vmse = f"{val_mse[i]:.6f}" if i < len(val_mse) else ""
                tacc = f"{train_acc[i]:.2f}" if i < len(train_acc) else ""
                vacc = f"{val_acc[i]:.2f}" if i < len(val_acc) else ""
                writer.writerow([e, tmse, vmse, tacc, vacc])

        return filepath

    def _export_predictions(self, output_dir: str, eval_results: dict) -> tuple[str, str]:
        """Gera os arquivos saidas_teste_continuas.csv e saidas_teste_discretas.csv."""
        continuous_path = os.path.join(output_dir, "saidas_teste_continuas.csv")
        discrete_path = os.path.join(output_dir, "saidas_teste_discretas.csv")

        labels = eval_results.get("class_labels", [chr(ord("A") + i) for i in range(26)])
        continuous_rows = eval_results.get("continuous_predictions", [])
        discrete_rows = eval_results.get("discrete_predictions", [])

        header_base = ["amostra", "classe_real", "classe_predita", "correto"]
        header_outputs = [f"y_{c}" for c in labels]
        full_header = header_base + header_outputs

        # 1. Saidas Continuas
        with open(continuous_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(full_header)
            for row in continuous_rows:
                vals = [f"{v:.6f}" for v in row["output"]]
                writer.writerow([row["sample_idx"], row["actual_label"], row["pred_label"], row["correct"]] + vals)

        # 2. Saidas Discretas
        with open(discrete_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(full_header)
            for row in discrete_rows:
                vals = [f"{int(v)}" if isinstance(v, (int, float)) and v.is_integer() else f"{v}" for v in row["output"]]
                writer.writerow([row["sample_idx"], row["actual_label"], row["pred_label"], row["correct"]] + vals)

        return continuous_path, discrete_path

    def export_official_deliverables(
        self, eval_results: dict, output_dir: str | None = None
    ) -> str:
        """
        Exporta os 5 arquivos oficiais exigidos para a entrega da disciplina na pasta saidas_oficiais/:
        1. hiperparametros.txt
        2. pesos_iniciais.csv (+ pesos_iniciais_V.csv, pesos_iniciais_W.csv)
        3. pesos_finais.csv (+ pesos_finais_V.csv, pesos_finais_W.csv)
        4. historico_erro_treinamento.csv
        5. saidas_teste_continuas.csv e saidas_teste_discretas.csv
        """
        target_dir = output_dir if output_dir is not None else os.path.join(self.output_dir, "saidas_oficiais")
        os.makedirs(target_dir, exist_ok=True)

        self._export_hyperparameters(target_dir, eval_results)
        self._export_weights(target_dir)
        self._export_history(target_dir)
        self._export_predictions(target_dir, eval_results)

        print(f"\nArquivos oficiais de entrega gerados em: {target_dir}")
        print("  - hiperparametros.txt")
        print("  - pesos_iniciais.csv (+ pesos_iniciais_V.csv, pesos_iniciais_W.csv)")
        print("  - pesos_finais.csv (+ pesos_finais_V.csv, pesos_finais_W.csv)")
        print("  - historico_erro_treinamento.csv")
        print("  - saidas_teste_continuas.csv")
        print("  - saidas_teste_discretas.csv")

        return target_dir


if __name__ == "__main__":
    model_arg = os.environ.get("MODEL")
    if not model_arg and len(sys.argv) > 1:
        model_arg = sys.argv[1]

    if not model_arg:
        print("\nErro: Nenhum modelo .json informado para avaliacao.")
        print("Uso: make evaluate MODEL=caminho/do/modelo.json")
        print(" ou: python3 -m lab.evaluator caminho/do/modelo.json\n")
        sys.exit(1)

    evaluator = MLPEvaluator(model_arg)
    results = evaluator.evaluate_test_set()
    evaluator.print_evaluation_report(results)
    evaluator.copy_model_to_results()
    evaluator.save_evaluation_report(results)
    evaluator.plot_learning_curves()
    evaluator.plot_confusion_matrix(results)
    evaluator.export_official_deliverables(results)
