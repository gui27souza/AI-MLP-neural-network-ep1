def build_report_text(model_name: str, eval_results: dict) -> str:
    """Gera o texto estruturado do relatorio de avaliacao."""
    class_acc = eval_results["class_accuracies"]
    confusions = eval_results["top_confusions"]
    sorted_by_acc = sorted(class_acc.items(), key=lambda x: (x[1]["accuracy"], x[0]))

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
    for label, data in sorted_by_acc[:5]:
        lines.append(f"  Letra '{label}': {data['accuracy']:>6.2f}% ({data['correct']}/{data['total']} acertos)")

    lines.extend(["", "[TOP 5 CARACTERES MAIS RECONHECIDOS (Maior Acuracia)]"])
    for label, data in reversed(sorted_by_acc[-5:]):
        lines.append(f"  Letra '{label}': {data['accuracy']:>6.2f}% ({data['correct']}/{data['total']} acertos)")

    lines.append("")
    if confusions:
        lines.append("[MAIORES CONFUSOES OBSERVADAS (Real -> Predito)]")
        for (actual, pred), count in confusions[:10]:
            lines.append(f"  Real '{actual}' classificado como '{pred}': {count} ocorrencia(s)")
    else:
        lines.append("Nenhuma confusao observada (100% de acertos).")

    lines.append("=" * 78)
    return "\n".join(lines)
