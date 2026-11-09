import os
import json
from datetime import datetime

from src.models import (
    MLP,
    Layer,
)


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
            "num_inputs": mlp.num_inputs,
            "num_outputs": mlp.num_outputs,
            "learn_rate": mlp.learn_rate,
            "momentum": mlp.momentum,
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

        mlp = MLP(
            num_inputs=data["num_inputs"],
            num_outputs=data["num_outputs"],
            learn_rate=data["learn_rate"],
            momentum=data.get("momentum", data.get("hyperparameters", {}).get("momentum", 0.0)),
            hidden_layers=hidden_layers,
            output_activation=activation_func,
            output_deriv_activation=derivative_activation,
        )

        # Sobrescreve os pesos e bias da camada de saída com os salvos
        mlp.output_layer.weights = data["output_layer"]["weights"]
        mlp.output_layer.bias = data["output_layer"]["bias"]

        print(f"Modelo carregado com sucesso de: {filepath}")

        return mlp
