from collections import defaultdict
from src.dataset_loader import DatasetLoader
from src.models import MLP


def evaluate_test_set(mlp: MLP, val_ratio: float = 0.3) -> dict:
    """
    Executa a inferencia no conjunto de teste e calcula a matriz de confusao,
    metricas de acuracia por classe e vetores de predicao continuos e discretos.
    """
    _, _, test_x, test_y, class_labels = DatasetLoader.load_caracteres_completo(val_ratio=val_ratio)

    num_samples = len(test_x)
    num_classes = len(class_labels)

    # Matriz de confusao [classe_real][classe_predita]
    confusion_matrix = [[0] * num_classes for _ in range(num_classes)]

    class_total = defaultdict(int)
    class_correct = defaultdict(int)
    confusions = defaultdict(int)

    total_correct = 0
    total_squared_error = 0.0

    continuous_rows = []
    discrete_rows = []

    for sample_idx, (x, target) in enumerate(zip(test_x, test_y)):
        output = mlp.predict(x)

        actual_idx = max(range(len(target)), key=lambda i: target[i])
        pred_idx = max(range(len(output)), key=lambda i: output[i])

        actual_label = class_labels[actual_idx]
        pred_label = class_labels[pred_idx]

        confusion_matrix[actual_idx][pred_idx] += 1
        class_total[actual_label] += 1

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
        "val_ratio": val_ratio,
        "continuous_predictions": continuous_rows,
        "discrete_predictions": discrete_rows,
    }


def build_report_text(model_name: str, eval_results: dict) -> str:
    """Gera o texto estruturado do relatorio de avaliacao."""
    class_acc = eval_results["class_accuracies"]
    confusions = eval_results["top_confusions"]

    lines = [
        "=" * 78,
        f"  RELATORIO DE AVALIACAO DO MODELO: {model_name}",
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
