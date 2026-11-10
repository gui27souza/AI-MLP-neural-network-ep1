import json
import os
import shutil
import sys
from collections import defaultdict

from src.dataset_loader import DatasetLoader
from src.funcs import sigmoide_bipolar, deriv_sigmoide_bipolar
from src.mlp_persister import MLPPersister
from src.models import MLP


class MLPEvaluator:
    """Responsável por avaliar modelos persistidos, gerar métricas detalhadas e gráficos de treinamento."""

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Arquivo de modelo não encontrado: {model_path}")

        self.model_path = model_path
        self.model_dir = os.path.dirname(os.path.abspath(model_path))
        self.model_filename = os.path.basename(model_path)
        self.model_name = os.path.splitext(self.model_filename)[0]

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

        for x, target in zip(test_x, test_y):
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

            if actual_idx == pred_idx:
                total_correct += 1
                class_correct[actual_label] += 1
            else:
                confusions[(actual_label, pred_label)] += 1

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
        target_dir = output_dir if output_dir is not None else os.path.join("results", self.model_name)
        os.makedirs(target_dir, exist_ok=True)
        report_path = os.path.join(target_dir, "evaluation_report.txt")

        report_text = self._build_report_text(eval_results)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text + "\n")

        print(f"Relatorio textual salvo em: {report_path}")
        return report_path

    def copy_model_to_results(self, output_dir: str | None = None) -> str:
        """Copia o arquivo .json do modelo para centralizar todos os artefatos de entrega."""
        target_dir = output_dir if output_dir is not None else os.path.join("results", self.model_name)
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

        target_dir = output_dir if output_dir is not None else os.path.join("results", self.model_name)
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

        target_dir = output_dir if output_dir is not None else os.path.join("results", self.model_name)
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
