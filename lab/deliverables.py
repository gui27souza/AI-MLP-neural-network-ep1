import csv
import os
import numpy as np

from src.mlp_persister import MLPPersister
from src.models import MLP


def export_hyperparameters(
    raw_data: dict,
    mlp: MLP,
    eval_results: dict | None,
    output_dir: str,
) -> str:
    """Gera o arquivo hiperparametros.txt com todos os parametros da rede e resultados."""
    filepath = os.path.join(output_dir, "hiperparametros.txt")

    hp = raw_data.get("hyperparameters", {})
    metrics = raw_data.get("metrics", {})
    training_params = metrics.get("training_params", {})
    stopping_info = metrics.get("stopping_info", {})
    ds_info = metrics.get("dataset_info", {})
    history = metrics.get("history", {})

    learn_rate = hp.get("learn_rate", getattr(mlp, "learn_rate", "N/A"))
    momentum = hp.get("momentum", getattr(mlp, "momentum", "N/A"))
    num_inputs = hp.get("num_inputs", mlp.num_inputs)
    num_outputs = hp.get("num_outputs", mlp.num_outputs)
    hidden_neurons = hp.get("hidden_neurons", [layer.num_neurons for layer in mlp.hidden_layers])

    max_epochs = training_params.get("max_epochs", "N/A")
    target_mse = training_params.get("target_mse", "N/A")
    patience = training_params.get("patience", "N/A")
    epochs_executed = metrics.get("epochs_executed", len(history.get("train_mse", [])))
    duration = metrics.get("duration_seconds", "N/A")

    train_samples = ds_info.get("train_samples", "N/A")
    val_samples = ds_info.get("val_samples", "N/A")
    val_ratio = eval_results.get("val_ratio", "N/A") if eval_results else "N/A"

    train_mse = metrics.get("train_mse", "N/A")
    train_acc = metrics.get("train_acc", "N/A")
    val_mse = metrics.get("val_mse", "N/A")
    val_acc = metrics.get("val_acc", "N/A")

    test_mse = eval_results.get("overall_mse", "N/A") if eval_results else "N/A"
    test_acc = eval_results.get("overall_accuracy", "N/A") if eval_results else "N/A"

    duration_str = f"{duration:.2f}s" if isinstance(duration, (int, float)) else str(duration)
    train_mse_str = f"{train_mse:.6f}" if isinstance(train_mse, (int, float)) else str(train_mse)
    train_acc_str = f"{train_acc:.2f}%" if isinstance(train_acc, (int, float)) else str(train_acc)
    val_mse_str = f"{val_mse:.6f}" if isinstance(val_mse, (int, float)) else str(val_mse)
    val_acc_str = f"{val_acc:.2f}%" if isinstance(val_acc, (int, float)) else str(val_acc)
    test_mse_str = f"{test_mse:.6f}" if isinstance(test_mse, (int, float)) else str(test_mse)
    test_acc_str = f"{test_acc:.2f}%" if isinstance(test_acc, (int, float)) else str(test_acc)

    lines = [
        "=" * 78,
        "                    CONFIGURACAO DE HIPERPARAMETROS DO MODELO",
        "=" * 78,
        f"Data de Geracao      : {raw_data.get('created_at', 'N/A')}",
        f"Tempo de Treinamento : {duration_str}",
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
        f"  - MSE de Treinamento        : {train_mse_str}",
        f"  - Acuracia de Treinamento   : {train_acc_str}",
        f"  - MSE de Validacao          : {val_mse_str}",
        f"  - Acuracia de Validacao     : {val_acc_str}",
        f"  - MSE de Teste              : {test_mse_str}",
        f"  - Acuracia de Teste         : {test_acc_str}",
        "=" * 78,
    ]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return filepath


def export_weights(mlp: MLP, output_dir: str) -> dict[str, str]:
    """Gera os arquivos de pesos no formato matricial de Fausett (V e W)."""
    saved_paths = {}

    # 1. Pesos Finais
    final_v_mats, final_w_mat = MLPPersister.to_fausett_matrices(mlp.get_weights())

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
    if getattr(mlp, "initial_weights", None) is not None:
        init_v_mats, init_w_mat = MLPPersister.to_fausett_matrices(mlp.initial_weights)

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
            f.write("# Pesos iniciais nao encontrados no arquivo .json.\n")
        saved_paths["pesos_iniciais.csv"] = dummy_init_path

    return saved_paths


def export_history(history: dict, output_dir: str) -> str:
    """Gera o historico_erro_treinamento.csv com MSE e acuracia por epoca."""
    filepath = os.path.join(output_dir, "historico_erro_treinamento.csv")
    train_mse = history.get("train_mse", [])
    val_mse = history.get("val_mse", [])
    train_acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])

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


def export_predictions(eval_results: dict, output_dir: str) -> tuple[str, str]:
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
    raw_data: dict,
    mlp: MLP,
    eval_results: dict,
    output_dir: str,
) -> str:
    """
    Exporta os 5 arquivos oficiais exigidos para a entrega da disciplina na pasta saidas_oficiais/:
    1. hiperparametros.txt
    2. pesos_iniciais.csv (+ pesos_iniciais_V.csv, pesos_iniciais_W.csv)
    3. pesos_finais.csv (+ pesos_finais_V.csv, pesos_finais_W.csv)
    4. historico_erro_treinamento.csv
    5. saidas_teste_continuas.csv e saidas_teste_discretas.csv
    """
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
