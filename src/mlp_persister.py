import os
import json
from datetime import datetime

from src.models import (
    MLP,
    Layer,
)
from src.orchestrator import Orchestrator


class MLPPersister:

    @staticmethod
    def _resolve_unique_filepath(filepath: str) -> str:
        """Se o arquivo já existir, adiciona sufixo incremental (_v2, _v3, ...) para não sobrescrever."""
        if not os.path.exists(filepath):
            return filepath

        base, ext = os.path.splitext(filepath)
        version = 2
        while os.path.exists(f"{base}_v{version}{ext}"):
            version += 1
        return f"{base}_v{version}{ext}"

    @staticmethod
    def save(mlp: MLP, filepath: str, metrics: dict | None = None) -> str:
        """Salva a estrutura, os pesos da rede e métricas opcionais em um arquivo JSON."""

        resolved_filepath = MLPPersister._resolve_unique_filepath(filepath)
        dirname = os.path.dirname(resolved_filepath)
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

        with open(resolved_filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        print(f"Modelo salvo com sucesso em: {resolved_filepath}")
        return resolved_filepath

    @staticmethod
    def load(
        filepath: str,
        activation_func,
        derivative_activation,
    ) -> "MLP":
        """Carrega uma MLP a partir de um arquivo JSON salvo previamente."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Reconstrói as camadas escondidas com os pesos salvos
        hidden_layers = []
        for h_data in data["hidden_layers"]:
            layer = Layer(
                num_inputs=h_data["num_inputs"],
                num_neurons=h_data["num_neurons"],
                activation_func=activation_func,
                derivative_activation=derivative_activation,
            )
            layer.weights = h_data["weights"]
            layer.bias = h_data["bias"]
            hidden_layers.append(layer)

        hyp = data.get("hyperparameters", {})
        num_inputs = hyp.get("num_inputs", data.get("num_inputs"))
        num_outputs = hyp.get("num_outputs", data.get("num_outputs"))
        learn_rate = hyp.get("learn_rate", data.get("learn_rate"))
        momentum = hyp.get("momentum", data.get("momentum", 0.0))

        mlp = MLP(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            learn_rate=learn_rate,
            momentum=momentum,
            hidden_layers=hidden_layers,
            output_activation=activation_func,
            output_deriv_activation=derivative_activation,
        )

        # Sobrescreve os pesos e bias da camada de saída com os salvos
        mlp.output_layer.weights = data["output_layer"]["weights"]
        mlp.output_layer.bias = data["output_layer"]["bias"]

        if "initial_weights" in data and data["initial_weights"]:
            mlp.initial_weights = data["initial_weights"]
        else:
            mlp.initial_weights = None

        print(f"Modelo carregado com sucesso de: {filepath}")

        return mlp

    @classmethod
    def save_checkpoint(
        cls,
        orch: Orchestrator,
        dir_path: str = "models/",
        prefix: str = "mlp_caracteres",
    ) -> str | None:
        """Extrai métricas do orquestrador e salva o modelo persistido com nome semântico."""

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

            # Identifica os picos históricos de validação (melhor época antes de eventual overfitting)
            best_val_acc = 0.0
            best_val_acc_epoch = 0
            if orch.history["val_acc"]:
                best_val_acc = max(orch.history["val_acc"])
                best_val_acc_epoch = orch.history["val_acc"].index(best_val_acc) + 1

            best_val_mse = 0.0
            best_val_mse_epoch = 0
            if orch.history["val_mse"]:
                best_val_mse = min(orch.history["val_mse"])
                best_val_mse_epoch = orch.history["val_mse"].index(best_val_mse) + 1

            # Se os melhores pesos foram restaurados pelo Early Stopping / Validação
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
                    "stop_reason": orch.stop_reason or "Executou todas as épocas programadas.",
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
            filepath = os.path.join(dir_path, filename)

            return cls.save(mlp, filepath, metrics=metrics)

        except Exception as e:
            print(f"Não foi possível salvar o checkpoint do modelo: {e}")
            return None
