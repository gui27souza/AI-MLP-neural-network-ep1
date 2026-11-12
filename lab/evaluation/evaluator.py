import json
import os
import shutil
import sys

from lab.evaluation.deliverables import export_official_deliverables
from lab.evaluation.metrics import evaluate_test_set
from lab.evaluation.plots import plot_confusion_matrix, plot_learning_curves
from lab.evaluation.reports import build_report_text
from src.core.funcs import deriv_sigmoide_bipolar, sigmoide_bipolar
from src.persistence.mlp_persister import MLPPersister


class MLPEvaluator:
    """Orquestrador de avaliacao de modelos persistidos, relatorios, graficos e entregas oficiais."""

    @staticmethod
    def _resolve_unique_dir(dir_path: str) -> str:
        """Adiciona sufixo incremental (_v2, _v3, ...) caso o diretorio ja contenha arquivos."""
        if not os.path.exists(dir_path) or not os.listdir(dir_path):
            return dir_path

        version = 2
        while os.path.exists(f"{dir_path}_v{version}"):
            if not os.listdir(f"{dir_path}_v{version}"):
                return f"{dir_path}_v{version}"
            version += 1
        return f"{dir_path}_v{version}"

    def _resolve_output_dir(self, output_dir: str | None = None) -> str:
        if output_dir is not None:
            return self._resolve_unique_dir(output_dir)

        expected_dir = os.path.abspath(os.path.join("results", self.model_name))
        current_dir = os.path.abspath(self.model_dir)
        if current_dir == expected_dir or current_dir.startswith(expected_dir + os.sep):
            return self.model_dir

        return self._resolve_unique_dir(os.path.join("results", self.model_name))

    def __init__(self, model_path: str, output_dir: str | None = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Arquivo de modelo nao encontrado: {model_path}")

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
        ds_info = self.metrics.get("dataset_info", {})
        train_samples = ds_info.get("train_samples", 0)
        val_samples = ds_info.get("val_samples", 0)
        total = train_samples + val_samples
        if total > 0 and val_samples > 0:
            return round(val_samples / total, 2)
        return 0.3

    def evaluate_test_set(self, val_ratio: float | None = None) -> dict:
        ratio = val_ratio if val_ratio is not None else self.get_val_ratio()
        return evaluate_test_set(self.mlp, val_ratio=ratio)

    def print_evaluation_report(self, eval_results: dict) -> None:
        print("\n" + build_report_text(self.model_filename, eval_results))

    def save_evaluation_report(self, eval_results: dict, output_dir: str | None = None) -> str:
        target_dir = output_dir if output_dir is not None else self.output_dir
        os.makedirs(target_dir, exist_ok=True)
        report_path = os.path.join(target_dir, "evaluation_report.txt")

        report_text = build_report_text(self.model_filename, eval_results)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text + "\n")

        print(f"Relatorio textual salvo em: {report_path}")
        return report_path

    def copy_model_to_results(self, output_dir: str | None = None) -> str:
        target_dir = output_dir if output_dir is not None else self.output_dir
        os.makedirs(target_dir, exist_ok=True)
        dest_path = os.path.join(target_dir, self.model_filename)

        if os.path.abspath(self.model_path) != os.path.abspath(dest_path):
            shutil.copy2(self.model_path, dest_path)
            print(f"Modelo .json copiado para a pasta de resultados: {dest_path}")

        return dest_path

    def plot_learning_curves(self, output_dir: str | None = None) -> str | None:
        target_dir = output_dir if output_dir is not None else self.output_dir
        stopping_info = self.metrics.get("stopping_info", {})
        return plot_learning_curves(self.history, stopping_info, self.model_filename, target_dir)

    def plot_confusion_matrix(self, eval_results: dict, output_dir: str | None = None) -> str | None:
        target_dir = output_dir if output_dir is not None else self.output_dir
        return plot_confusion_matrix(eval_results, self.model_filename, target_dir)

    def export_official_deliverables(self, eval_results: dict, output_dir: str | None = None) -> str:
        target_dir = output_dir if output_dir is not None else os.path.join(self.output_dir, "saidas_oficiais")
        return export_official_deliverables(self.raw_data, self.mlp, eval_results, target_dir)


if __name__ == "__main__":
    model_arg = os.environ.get("MODEL")
    if not model_arg and len(sys.argv) > 1:
        model_arg = sys.argv[1]

    if not model_arg:
        print("\nErro: Nenhum modelo .json informado para avaliacao.")
        print("Uso: make evaluate MODEL=caminho/do/modelo.json")
        print(" ou: python3 -m lab.evaluation.evaluator caminho/do/modelo.json\n")
        sys.exit(1)

    evaluator = MLPEvaluator(model_arg)
    results = evaluator.evaluate_test_set()
    evaluator.print_evaluation_report(results)
    evaluator.copy_model_to_results()
    evaluator.save_evaluation_report(results)
    evaluator.plot_learning_curves()
    evaluator.plot_confusion_matrix(results)
    evaluator.export_official_deliverables(results)
