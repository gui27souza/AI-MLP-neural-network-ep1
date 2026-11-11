import json
import os
from datetime import datetime

import numpy as np

from src.models import MLP, Layer
from src.orchestrator import Orchestrator


class MLPPersister:
    """
    Modulo responsavel pela persistencia e serializacao da rede MLP.

    Suporta:
    1. Salvamento e carregamento completo de arquitetura, pesos e historico em JSON.
    2. Checkpoints automaticos durante treino com nomenclatura semantica.
    3. Conversao bidirecional entre dicionarios internos e matrizes de Fausett (V e W).
    """

    # =========================================================================
    # 1. RESOLUCAO DE ARQUIVOS E PERSISTENCIA JSON
    # =========================================================================

    @staticmethod
    def _resolve_unique_filepath(filepath: str) -> str:
        """Evita sobrescrita adicionando sufixo incremental (_v2, _v3, ...) se o arquivo existir."""
        if not os.path.exists(filepath):
            return filepath

        base, ext = os.path.splitext(filepath)
        version = 2
        while os.path.exists(f"{base}_v{version}{ext}"):
            version += 1
        return f"{base}_v{version}{ext}"

    @classmethod
    def save(cls, mlp: MLP, filepath: str, metrics: dict | None = None) -> str:
        """Serializa a estrutura, os pesos da rede e metricas opcionais em um arquivo JSON."""
        resolved = cls._resolve_unique_filepath(filepath)
        dirname = os.path.dirname(resolved)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        data = {
            "created_at": datetime.now().isoformat(),
            "hyperparameters": {
                "learn_rate": mlp.learn_rate,
                "momentum": mlp.momentum,
                "num_inputs": mlp.num_inputs,
                "num_outputs": mlp.num_outputs,
                "hidden_neurons": [layer.num_neurons for layer in mlp.hidden_layers],
            },
            "metrics": metrics or {},
            "initial_weights": getattr(mlp, "initial_weights", None),
            "hidden_layers": [
                {
                    "num_inputs": layer.num_inputs,
                    "num_neurons": layer.num_neurons,
                    "weights": layer.weights,
                    "bias": layer.bias,
                }
                for layer in mlp.hidden_layers
            ],
            "output_layer": {
                "num_inputs": mlp.output_layer.num_inputs,
                "num_neurons": mlp.output_layer.num_neurons,
                "weights": mlp.output_layer.weights,
                "bias": mlp.output_layer.bias,
            },
        }

        with open(resolved, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        print(f"Modelo salvo com sucesso em: {resolved}")
        return resolved

    @classmethod
    def load(cls, filepath: str, activation_func, derivative_activation) -> MLP:
        """Reconstroi uma MLP e seus parametros a partir de um arquivo JSON persistido."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        hidden_layers = [
            Layer(
                num_inputs=h["num_inputs"],
                num_neurons=h["num_neurons"],
                activation_func=activation_func,
                derivative_activation=derivative_activation,
            )
            for h in data["hidden_layers"]
        ]
        for layer, h_data in zip(hidden_layers, data["hidden_layers"]):
            layer.weights = h_data["weights"]
            layer.bias = h_data["bias"]

        hyp = data.get("hyperparameters", {})
        mlp = MLP(
            num_inputs=hyp.get("num_inputs", data.get("num_inputs")),
            num_outputs=hyp.get("num_outputs", data.get("num_outputs")),
            learn_rate=hyp.get("learn_rate", data.get("learn_rate")),
            momentum=hyp.get("momentum", data.get("momentum", 0.0)),
            hidden_layers=hidden_layers,
            output_activation=activation_func,
            output_deriv_activation=derivative_activation,
        )

        mlp.output_layer.weights = data["output_layer"]["weights"]
        mlp.output_layer.bias = data["output_layer"]["bias"]
        mlp.initial_weights = data.get("initial_weights")

        print(f"Modelo carregado com sucesso de: {filepath}")
        return mlp

    # =========================================================================
    # 2. CHECKPOINT DE TREINAMENTO
    # =========================================================================

    @classmethod
    def save_checkpoint(
        cls,
        orch: Orchestrator,
        dir_path: str = "models/",
        prefix: str = "mlp_caracteres",
    ) -> str | None:
        """Extrai metricas consolidadas do orquestrador e salva o modelo com nome semantico."""
        try:
            mlp = orch.mlp
            duration = getattr(orch, "delta_ts", None)
            if duration is None and hasattr(orch, "start_ts"):
                duration = datetime.now().timestamp() - orch.start_ts
            else:
                duration = duration or 0.0

            val_acc_pct = (orch.history["val_acc"][-1] * 100) if orch.history["val_acc"] else 0.0
            train_acc_pct = (orch.history["train_acc"][-1] * 100) if orch.history["train_acc"] else 0.0
            val_mse = orch.history["val_mse"][-1] if orch.history["val_mse"] else 0.0

            best_val_acc = max(orch.history["val_acc"]) if orch.history["val_acc"] else 0.0
            best_val_acc_epoch = orch.history["val_acc"].index(best_val_acc) + 1 if orch.history["val_acc"] else 0

            best_val_mse = min(orch.history["val_mse"]) if orch.history["val_mse"] else 0.0
            best_val_mse_epoch = orch.history["val_mse"].index(best_val_mse) + 1 if orch.history["val_mse"] else 0

            weights_restored = orch.best_weights is not None
            saved_epoch = orch.best_epoch if weights_restored else orch.total_epochs
            saved_val_mse = orch.best_val_mse if weights_restored else val_mse
            saved_val_acc_pct = (orch.best_val_acc * 100) if weights_restored else val_acc_pct

            metrics = {
                "dataset_info": {
                    "train_samples": len(orch.inputs),
                    "val_samples": len(orch.val_inputs) if orch.val_inputs else 0,
                },
                "training_params": {
                    "max_epochs": orch.epochs,
                    "target_mse": orch.target_mse,
                    "patience": orch.patience,
                },
                "stopping_info": {
                    "stopped_early": orch.stopped_early,
                    "stop_reason": orch.stop_reason or "Executou todas as epocas programadas.",
                    "weights_restored_from_best": weights_restored,
                    "saved_epoch": saved_epoch,
                },
                "epochs_executed": orch.total_epochs,
                "duration_seconds": round(duration, 2),
                "train_mse": orch.history["train_mse"][-1] if orch.history["train_mse"] else None,
                "train_acc": round(train_acc_pct, 2),
                "val_mse": round(val_mse, 5),
                "val_acc": round(val_acc_pct, 2),
                "best_val_acc": round(best_val_acc * 100, 2),
                "best_val_acc_epoch": best_val_acc_epoch,
                "best_val_mse": round(best_val_mse, 5),
                "best_val_mse_epoch": best_val_mse_epoch,
                "history": {
                    "train_mse": [round(x, 6) for x in orch.history.get("train_mse", [])],
                    "train_acc": [round(x * 100, 2) for x in orch.history.get("train_acc", [])],
                    "val_mse": [round(x, 6) for x in orch.history.get("val_mse", [])],
                    "val_acc": [round(x * 100, 2) for x in orch.history.get("val_acc", [])],
                },
            }

            hidden_desc = "h" + "_".join(str(layer.num_neurons) for layer in mlp.hidden_layers)
            filename = (
                f"{prefix}_{hidden_desc}"
                f"_acc{saved_val_acc_pct:.1f}_mse{saved_val_mse:.3f}"
                f"_e{saved_epoch}.json"
            )
            return cls.save(mlp, os.path.join(dir_path, filename), metrics=metrics)

        except Exception as e:
            print(f"Nao foi possivel salvar o checkpoint do modelo: {e}")
            return None

    # =========================================================================
    # 3. FORMATO MATRICIAL DE FAUSSETT (V e W)
    # =========================================================================

    @staticmethod
    def to_fausett_matrices(weights_state: dict) -> tuple[list[np.ndarray], np.ndarray]:
        """
        Converte o dicionario de pesos para a convencao de matrizes de Laurene Fausett:
        - Para cada camada oculta j: matriz V com dimensao (p, n + 1),
          onde coluna 0 = bias (v_0j) e colunas 1..n = pesos (v_ij).
        - Para a camada de saida k: matriz W com dimensao (m, p + 1),
          onde coluna 0 = bias (w_0k) e colunas 1..p = pesos (w_jk).
        """
        hidden_matrices = []
        for layer_state in weights_state["hidden_layers"]:
            rows = [[b] + list(w) for b, w in zip(layer_state["bias"], layer_state["weights"])]
            hidden_matrices.append(np.array(rows, dtype=float))

        out_rows = [[b] + list(w) for b, w in zip(weights_state["output_layer"]["bias"], weights_state["output_layer"]["weights"])]
        output_matrix = np.array(out_rows, dtype=float)

        return hidden_matrices, output_matrix

    @staticmethod
    def from_fausett_matrices(
        hidden_matrices: list[np.ndarray] | np.ndarray,
        output_matrix: np.ndarray,
    ) -> dict:
        """
        Reconstroi o dicionario de pesos a partir das matrizes V e W de Fausett:
        - Coluna 0: bias (v_0j ou w_0k).
        - Colunas 1..fim: pesos de conexao (v_ij ou w_jk).
        """
        if isinstance(hidden_matrices, np.ndarray) and hidden_matrices.ndim == 2:
            hidden_matrices = [hidden_matrices]

        hidden_layers_state = []
        for v_mat in hidden_matrices:
            v_arr = np.array(v_mat, dtype=float)
            hidden_layers_state.append({
                "bias": v_arr[:, 0].tolist(),
                "weights": v_arr[:, 1:].tolist(),
            })

        out_arr = np.array(output_matrix, dtype=float)
        return {
            "hidden_layers": hidden_layers_state,
            "output_layer": {
                "bias": out_arr[:, 0].tolist(),
                "weights": out_arr[:, 1:].tolist(),
            },
        }

    @staticmethod
    def load_initial_weights(filepath: str) -> dict:
        """Carrega os pesos iniciais salvos de um arquivo JSON de checkpoint."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "initial_weights" in data and data["initial_weights"]:
            return data["initial_weights"]
        if "hidden_layers" in data and "output_layer" in data:
            return {
                "hidden_layers": data["hidden_layers"],
                "output_layer": data["output_layer"],
            }
        raise ValueError(f"Estrutura de pesos invalida no arquivo JSON: {filepath}")
