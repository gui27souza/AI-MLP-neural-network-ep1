import os
import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(
    history: dict,
    stopping_info: dict,
    model_name: str,
    output_dir: str,
) -> str | None:
    """Gera o grafico comparativo de curvas de aprendizado (MSE e Acuracia)."""
    train_mse = history.get("train_mse", [])
    val_mse = history.get("val_mse", [])
    train_acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])

    if not train_mse:
        return None

    os.makedirs(output_dir, exist_ok=True)
    epochs = list(range(1, len(train_mse) + 1))
    saved_epoch = stopping_info.get("saved_epoch")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Curva de MSE
    ax1.plot(epochs, train_mse, label="MSE Treino", color="#1f77b4", linewidth=2)
    if val_mse:
        ax1.plot(epochs, val_mse, label="MSE Validacao", color="#ff7f0e", linewidth=2, linestyle="--")
    if saved_epoch and saved_epoch <= len(epochs):
        ax1.axvline(x=saved_epoch, color="#2ca02c", linestyle=":", label=f"Melhor Epoca ({saved_epoch})")
    ax1.set_title("Curva de Aprendizado - MSE")
    ax1.set_xlabel("Epocas")
    ax1.set_ylabel("MSE")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # 2. Curva de Acuracia
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

    plt.suptitle(f"Evolucao do Treinamento - {model_name}", fontsize=13)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "learning_curves.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"\nGrafico de curvas de aprendizado salvo em: {output_path}")
    return output_path


def plot_confusion_matrix(eval_results: dict, model_name: str, output_dir: str) -> str | None:
    """Gera mapa de calor da matriz de confusao com contagem por celula."""
    os.makedirs(output_dir, exist_ok=True)

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
        f"Matriz de Confusao - {model_name}\nAcuracia Geral: {eval_results['overall_accuracy']:.2f}%",
        fontsize=13,
        pad=15,
    )

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix[i, j]
            if val > 0:
                color = "white" if val > matrix.max() / 2 else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Matriz de confusao salva em: {output_path}")
    return output_path
