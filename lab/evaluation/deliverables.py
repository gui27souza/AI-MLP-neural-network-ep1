import csv
import os
import numpy as np

from src.core.models import MLP
from src.persistence.mlp_persister import MLPPersister


def _fmt(val, is_pct: bool = False, precision: int = 6) -> str:
    if not isinstance(val, (int, float)):
        return str(val)
    return f"{val:.2f}%" if is_pct else f"{val:.{precision}f}"


def export_hyperparameters(
    raw_data: dict,
    mlp: MLP,
    eval_results: dict | None,
    output_dir: str,
) -> str:
    """Exporta relatorio textual com configuracao e metricas do modelo."""
    filepath = os.path.join(output_dir, "hiperparametros.txt")

    hp = raw_data.get("hyperparameters", {})
    metrics = raw_data.get("metrics", {})
    train_params = metrics.get("training_params", {})
    stop_info = metrics.get("stopping_info", {})
    ds_info = metrics.get("dataset_info", {})

    hidden_neurons = hp.get("hidden_neurons", [layer.num_neurons for layer in mlp.hidden_layers])
    duration = metrics.get("duration_seconds", "N/A")
    duration_str = f"{duration:.2f}s" if isinstance(duration, (int, float)) else str(duration)

    val_ratio = eval_results.get("val_ratio", "N/A") if eval_results else "N/A"
    test_mse = eval_results.get("overall_mse", "N/A") if eval_results else "N/A"
    test_acc = eval_results.get("overall_accuracy", "N/A") if eval_results else "N/A"

    lines = [
        "=" * 78,
        "                    CONFIGURACAO DE HIPERPARAMETROS DO MODELO",
        "=" * 78,
        f"Data de Geracao      : {raw_data.get('created_at', 'N/A')}",
        f"Tempo de Treinamento : {duration_str}",
        "",
        "[1. ARQUITETURA DA REDE]",
        f"  - Entradas (n)              : {hp.get('num_inputs', mlp.num_inputs)} (caracteres 10x12 normalizados [-1, 1])",
        f"  - Camada(s) Oculta(s) (p)   : {hidden_neurons} neuronios",
        f"  - Saidas (m)                : {hp.get('num_outputs', mlp.num_outputs)} neuronios (classes A-Z)",
        "  - Funcao de Ativacao        : Sigmoide Bipolar (faixa [-1, 1])",
        "  - Derivada da Ativacao      : f'(x) = 0.5 * (1 + f(x)) * (1 - f(x))",
        "",
        "[2. HIPERPARAMETROS DE TREINAMENTO]",
        f"  - Taxa de Aprendizado (eta) : {hp.get('learn_rate', getattr(mlp, 'learn_rate', 'N/A'))}",
        f"  - Termo de Momento (alpha)  : {hp.get('momentum', getattr(mlp, 'momentum', 'N/A'))}",
        f"  - Maximo de Epocas          : {train_params.get('max_epochs', 'N/A')}",
        f"  - Epocas Executadas         : {metrics.get('epochs_executed', 'N/A')}",
        f"  - Criterio de Parada        : Early Stopping (Paciencia={train_params.get('patience', 'N/A')}, Min Delta MSE={train_params.get('target_mse', 'N/A')})",
        f"  - Parada Acionada           : {stop_info.get('stopped_early', False)}",
        f"  - Motivo da Parada          : {stop_info.get('stop_reason', 'N/A')}",
        f"  - Epoca do Melhor Modelo    : {stop_info.get('saved_epoch', 'N/A')}",
        "",
        "[3. DADOS E AMOSTRAGEM]",
        "  - Dataset                   : caracteres-completo (EACH-USP)",
        f"  - Amostras de Treinamento   : {ds_info.get('train_samples', 'N/A')}",
        f"  - Amostras de Validacao     : {ds_info.get('val_samples', 'N/A')} (Ratio={val_ratio})",
        f"  - Amostras de Teste         : {eval_results.get('num_samples', 'N/A') if eval_results else 'N/A'}",
        "",
        "[4. DESEMPENHO ALCANCADO]",
        f"  - MSE de Treinamento        : {_fmt(metrics.get('train_mse', 'N/A'))}",
        f"  - Acuracia de Treinamento   : {_fmt(metrics.get('train_acc', 'N/A'), is_pct=True)}",
        f"  - MSE de Validacao          : {_fmt(metrics.get('val_mse', 'N/A'))}",
        f"  - Acuracia de Validacao     : {_fmt(metrics.get('val_acc', 'N/A'), is_pct=True)}",
        f"  - MSE de Teste              : {_fmt(test_mse)}",
        f"  - Acuracia de Teste         : {_fmt(test_acc, is_pct=True)}",
        "=" * 78,
    ]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return filepath


def _export_weight_group(
    weights_dict: dict | None,
    prefix: str,
    output_dir: str,
    header_tag: str = "",
) -> dict[str, str]:
    """Helper unificado para exportar matrizes V e W individuais e combinadas."""
    paths: dict[str, str] = {}
    combined_name = f"pesos_{prefix}.csv"
    combined_path = os.path.join(output_dir, combined_name)

    if weights_dict is None:
        with open(combined_path, "w", encoding="utf-8") as f:
            f.write("# Pesos nao encontrados.\n")
        paths[combined_name] = combined_path
        return paths

    v_mats, w_mat = MLPPersister.to_fausett_matrices(weights_dict)

    for idx, v_mat in enumerate(v_mats):
        v_name = f"pesos_{prefix}_V.csv" if len(v_mats) == 1 else f"pesos_{prefix}_V{idx+1}.csv"
        v_path = os.path.join(output_dir, v_name)
        np.savetxt(v_path, v_mat, delimiter=",", fmt="%.8f")
        paths[v_name] = v_path

    w_name = f"pesos_{prefix}_W.csv"
    w_path = os.path.join(output_dir, w_name)
    np.savetxt(w_path, w_mat, delimiter=",", fmt="%.8f")
    paths[w_name] = w_path

    with open(combined_path, "w", encoding="utf-8") as f:
        tag = f" {header_tag}" if header_tag else ""
        for idx, v_mat in enumerate(v_mats):
            suffix = f" {idx+1}" if len(v_mats) > 1 else ""
            f.write(f"# MATRIZ V{suffix}{tag} (Camada Oculta: {v_mat.shape[0]} neuronios x {v_mat.shape[1]} colunas [col 0 = bias])\n")
            np.savetxt(f, v_mat, delimiter=",", fmt="%.8f")
            f.write("\n")
        f.write(f"# MATRIZ W{tag} (Camada de Saida: {w_mat.shape[0]} neuronios x {w_mat.shape[1]} colunas [col 0 = bias])\n")
        np.savetxt(f, w_mat, delimiter=",", fmt="%.8f")

    paths[combined_name] = combined_path
    return paths


def export_weights(mlp: MLP, output_dir: str) -> dict[str, str]:
    """Exporta matrizes de pesos finais e iniciais no padrao Fausett."""
    saved_paths = _export_weight_group(mlp.get_weights(), "finais", output_dir)
    initial_weights = getattr(mlp, "initial_weights", None)
    saved_paths.update(_export_weight_group(initial_weights, "iniciais", output_dir, header_tag="INICIAL"))
    return saved_paths


def export_history(history: dict, output_dir: str) -> str:
    """Exporta o historico de MSE e acuracia por epoca em CSV."""
    filepath = os.path.join(output_dir, "historico_erro_treinamento.csv")
    train_mse = history.get("train_mse", [])
    val_mse = history.get("val_mse", [])
    train_acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])

    num_epochs = max(len(train_mse), len(val_mse), len(train_acc), len(val_acc))
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoca", "mse_treinamento", "mse_validacao", "acuracia_treinamento", "acuracia_validacao"])
        for i in range(num_epochs):
            writer.writerow([
                i + 1,
                f"{train_mse[i]:.6f}" if i < len(train_mse) else "",
                f"{val_mse[i]:.6f}" if i < len(val_mse) else "",
                f"{train_acc[i]:.2f}" if i < len(train_acc) else "",
                f"{val_acc[i]:.2f}" if i < len(val_acc) else "",
            ])
    return filepath


def export_predictions(eval_results: dict, output_dir: str) -> tuple[str, str]:
    """Exporta predicoes continuas e discretizadas de teste em CSV."""
    labels = eval_results.get("class_labels", [chr(ord("A") + i) for i in range(26)])
    header = ["amostra", "classe_real", "classe_predita", "correto"] + [f"y_{c}" for c in labels]

    continuous_path = os.path.join(output_dir, "saidas_teste_continuas.csv")
    with open(continuous_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in eval_results.get("continuous_predictions", []):
            writer.writerow([r["sample_idx"], r["actual_label"], r["pred_label"], r["correct"]] + [f"{v:.6f}" for v in r["output"]])

    discrete_path = os.path.join(output_dir, "saidas_teste_discretas.csv")
    with open(discrete_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in eval_results.get("discrete_predictions", []):
            vals = [int(v) if isinstance(v, (int, float)) and v.is_integer() else v for v in r["output"]]
            writer.writerow([r["sample_idx"], r["actual_label"], r["pred_label"], r["correct"]] + vals)

    return continuous_path, discrete_path


def export_official_deliverables(
    raw_data: dict,
    mlp: MLP,
    eval_results: dict,
    output_dir: str,
) -> str:
    """Gera os arquivos oficiais de entrega em output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    export_hyperparameters(raw_data, mlp, eval_results, output_dir)
    export_weights(mlp, output_dir)
    export_history(raw_data.get("metrics", {}).get("history", {}), output_dir)
    export_predictions(eval_results, output_dir)

    print(f"\nArquivos oficiais de entrega gerados em: {output_dir}")
    print("  - hiperparametros.txt")
    print("  - pesos_iniciais.csv (+ pesos_iniciais_V.csv, pesos_iniciais_W.csv)")
    print("  - pesos_finais.csv (+ pesos_finais_V.csv, pesos_finais_W.csv)")
    print("  - historico_erro_treinamento.csv")
    print("  - saidas_teste_continuas.csv")
    print("  - saidas_teste_discretas.csv")
    return output_dir
