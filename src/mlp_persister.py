import os
import json

from src.models import (
    MLP,
    Layer,
)


class MLPPersister:

    def save(self, mlp: MLP, filepath: str) -> None:
        """Salva a estrutura e os pesos da rede em um arquivo JSON."""

        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        data = {
            "num_inputs": mlp.num_inputs,
            "num_outputs": mlp.num_outputs,
            "learn_rate": mlp.learn_rate,
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

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        print(f"Modelo salvo com sucesso em: {filepath}")

    def load(
        self,
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
            hidden_layers=hidden_layers,
            output_activation=activation_func,
            output_deriv_activation=derivative_activation,
        )

        # Sobrescreve os pesos e bias da camada de saída com os salvos
        mlp.output_layer.weights = data["output_layer"]["weights"]
        mlp.output_layer.bias = data["output_layer"]["bias"]

        print(f"Modelo carregado com sucesso de: {filepath}")

        return mlp
