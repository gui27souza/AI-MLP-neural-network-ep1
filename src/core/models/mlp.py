from typing import Callable
import numpy as np

from .layer import Layer


class MLP:
    """
    Rede Neural Artificial Multilayer Perceptron (MLP) — Backpropagation com GD (Fausett 6.1.2).

    Estrutura Arquitetural (Slide 2):
    - Camada de Entrada (n=120 pixels) -> Camada Escondida (p neurônios, matriz V)
    - Camada de Saída (m=26 classes 'A'..'Z', matriz W).

    Integrantes do Grupo (EACH-USP | Inteligência Artificial 2026):
    - Guilherme Oliveira de Souza (Nº USP: 14671388)
    - Paulo Rogerio Andrade de Oliveira (Nº USP: 9365783)
    - Victor Hugo Antunes Valadares (Nº USP: 14760317)

    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        learn_rate: float,
        hidden_layers: list[Layer],
        output_activation: Callable[[float], float],
        output_deriv_activation: Callable[[float], float],
        weight_range: tuple[float, float] = (-0.2, 0.2),
        bias_range: tuple[float, float] = (-0.2, 0.2),
        momentum: float = 0.0,
        initial_weights: dict | None = None,
    ):
        # Valida encadeamento de dimensões entre entrada, camada(s) escondida(s) e saída
        if hidden_layers:
            if num_inputs != hidden_layers[0].num_inputs:
                raise ValueError(
                    "Número de entradas deve ser igual ao da primeira camada escondida."
                )

            previous_num_outputs = num_inputs
            for i, layer in enumerate(hidden_layers):
                if layer.num_inputs != previous_num_outputs:
                    raise ValueError(
                        f"Número de entradas da camada {i + 1} incompatível "
                        "com o número de saídas da camada anterior. "
                        f"Entradas: {layer.num_inputs} | "
                        f"Saídas da camada anterior: {previous_num_outputs}"
                    )
                previous_num_outputs = layer.num_neurons

            output_layer_inputs = hidden_layers[-1].num_neurons
        else:
            output_layer_inputs = num_inputs

        self.output_layer = Layer(
            num_inputs=output_layer_inputs,
            num_neurons=num_outputs,
            activation_func=output_activation,
            derivative_activation=output_deriv_activation,
            weight_range=weight_range,
            bias_range=bias_range,
        )

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        self.learn_rate = learn_rate
        self.momentum = momentum

        # Configura os pesos iniciais (injetados para determinismo ou gerados aleatoriamente)
        if initial_weights is not None:
            self.set_initial_weights(initial_weights)
        else:
            self.initial_weights = self.get_weights()

    # =========================================================================
    #  FEEDFORWARD, CÁLCULO DO ERRO E RECONHECIMENTO (Slides 3 e 7)
    # =========================================================================

    def predict(self, inputs: list[float]) -> list[float]:
        """
        Executa a fase de Feedforward completa da rede (Passos 3, 4 e 5 de Fausett — Slide 3):
        Propaga o vetor de entrada x pela(s) camada(s) escondida(s) até a camada de saída.
        """
        if len(inputs) != self.num_inputs:
            raise ValueError(
                f"Número de entradas incompatível: esperado {self.num_inputs} e recebido {len(inputs)}"
            )

        current_activations = inputs
        for layer in self.hidden_layers:
            current_activations = layer.forward(current_activations)

        return self.output_layer.forward(current_activations)

    def predict_class(self, outputs: list[float]) -> int:
        """
        Procedimento de decisão para reconhecimento de caractere (Slide 7):
        Aplica a regra Winner-Takes-All (argmax), selecionando o índice do neurônio
        de saída k com maior valor de ativação contínua y_k.
        """
        return int(np.argmax(outputs))

    def recognize_character(
        self, inputs: list[float], class_labels: list[str] | None = None
    ) -> tuple[int, str, list[float]]:
        """
        Realiza a inferência para uma amostra de caractere e retorna (índice, letra, vetor_saída).
        """
        outputs = self.predict(inputs)
        pred_idx = self.predict_class(outputs)
        labels = class_labels or [chr(ord("A") + i) for i in range(self.num_outputs)]
        pred_char = labels[pred_idx] if pred_idx < len(labels) else str(pred_idx)
        return pred_idx, pred_char, outputs

    @staticmethod
    def compute_mse(outputs_arr: np.ndarray, targets_arr: np.ndarray) -> float:
        """
        Calcula o Erro Quadrático Médio (MSE) na saída da rede neural (Slide 3 e Slide 11):
        MSE = média de (t_k - y_k)^2 sobre todas as amostras e neurônios de saída.
        """
        return float(np.mean((targets_arr - outputs_arr) ** 2))

    def compute_accuracy(
        self, outputs_arr: np.ndarray, targets_arr: np.ndarray
    ) -> float:
        """
        Calcula a taxa de acerto (acurácia) das predições da rede:
        - Multiclasse (num_outputs > 1): compara o neurônio vencedor (argmax) com o alvo.
        - Binário (num_outputs == 1): aplica limiar de sinal (>= 0 -> +1.0, < 0 -> -1.0).
        """
        if self.num_outputs > 1:
            preds = np.argmax(outputs_arr, axis=1)
            actuals = np.argmax(targets_arr, axis=1)
            return float(np.mean(preds == actuals))

        preds = np.where(outputs_arr >= 0, 1.0, -1.0)
        return float(np.mean(preds == targets_arr))

    def evaluate(
        self, inputs_list: list[list[float]], targets: list[list[float]]
    ) -> tuple[float, float]:
        """
        Avalia um conjunto de dados retornando (MSE, Acurácia) sem alterar os pesos da rede.
        """
        outputs = [self.predict(x) for x in inputs_list]
        outputs_arr = np.array(outputs)
        targets_arr = np.array(targets)

        mse = self.compute_mse(outputs_arr, targets_arr)
        acc = self.compute_accuracy(outputs_arr, targets_arr)
        return mse, acc

    # =========================================================================
    #  TREINAMENTO ONLINE E BACKPROPAGATION (Slides 4, 5 e 6)
    # =========================================================================

    def run_epoch(
        self,
        inputs_list: list[list[float]],
        targets: list[list[float]],
        print_info: bool = False,
    ) -> None:
        """
        Executa uma época completa de treinamento apresentando cada par (x, t) da base.
        """
        if len(inputs_list) != len(targets):
            raise ValueError(
                "Número de entradas deve ser igual ao de targets: "
                f"entradas {len(inputs_list)} e targets {len(targets)}"
            )

        for input_data, target in zip(inputs_list, targets):
            output = self._run_iteration(input_data, target)
            if print_info:
                print(f"in: {input_data}\nout:    {output}\ntarget: {target}\n")

    def _run_iteration(self, inputs: list[float], target: list[float]) -> list[float]:
        """
        Executa uma iteração de aprendizado online para um único padrão (x, t):
        1. Propagação direta (Feedforward)
        2. Retropropagação do erro e ajuste de pesos (Backpropagation)
        """
        if len(target) != self.num_outputs:
            raise ValueError(
                f"Tamanho de target incompatível: esperado {self.num_outputs} e recebido {len(target)}"
            )

        outputs = self.predict(inputs)
        self._update_weights(target, outputs)
        return outputs

    def _compute_output_deltas(
        self, target: list[float], outputs: list[float]
    ) -> list[float]:
        """
        Passo 6 de Fausett (Slide 4) — Erro na camada de saída:
        δ_k = (t_k - y_k) * f'(y_in_k)
        """
        return self.output_layer.compute_output_deltas(target, outputs)

    def _compute_hidden_deltas(
        self, output_deltas: list[float]
    ) -> list[list[float]]:
        """
        Passo 7 de Fausett (Slide 5) — Contribuição do erro na(s) camada(s) escondida(s):
        δ_in_j = Σ_k (δ_k * w_jk)  =>  δ_j = δ_in_j * f'(z_in_j)
        Calculado usando os pesos atuais (antes da atualização do Passo 8).
        """
        hidden_deltas_per_layer: list[list[float]] = []
        next_deltas = output_deltas
        next_weights = self.output_layer.weights

        for layer in reversed(self.hidden_layers):
            current_deltas = layer.compute_hidden_deltas(
                next_layer_deltas=next_deltas,
                next_layer_weights=next_weights,
            )
            hidden_deltas_per_layer.insert(0, current_deltas)
            next_deltas = current_deltas
            next_weights = layer.weights

        return hidden_deltas_per_layer

    def _apply_weight_updates(
        self,
        output_deltas: list[float],
        hidden_deltas_per_layer: list[list[float]],
    ) -> None:
        """
        Passo 8 de Fausett (Slide 6) — Atualização dos pesos sinápticos e biases:
        Aplica a regra Delta (Gradiente Descendente + Momentum opcional) na camada
        de saída (matriz W) e na(s) camada(s) escondida(s) (matriz V).
        """
        self.output_layer.update_weights(
            deltas=output_deltas,
            learn_rate=self.learn_rate,
            momentum=self.momentum,
        )

        for layer, hidden_deltas in zip(self.hidden_layers, hidden_deltas_per_layer):
            layer.update_weights(
                deltas=hidden_deltas,
                learn_rate=self.learn_rate,
                momentum=self.momentum,
            )

    def _update_weights(self, target: list[float], outputs: list[float]) -> None:
        """
        Orquestra a fase de Backpropagation em 3 etapas modulares (Passos 6, 7 e 8 de Fausett):
        1. Calcula informação de erro na camada de saída (δ_k)
        2. Calcula contribuição do erro na camada escondida (δ_j)
        3. Atualiza pesos e biases de todas as camadas (ΔW e ΔV)
        """
        output_deltas = self._compute_output_deltas(target, outputs)
        hidden_deltas_per_layer = self._compute_hidden_deltas(output_deltas)
        self._apply_weight_updates(output_deltas, hidden_deltas_per_layer)

    # =========================================================================
    #  GERENCIAMENTO DE ESTADO E PESOS INICIAIS/FINAIS
    # =========================================================================

    def get_weights(self) -> dict:
        """Retorna uma cópia dos pesos e biases de todas as camadas da rede."""
        return {
            "hidden_layers": [layer.get_weights() for layer in self.hidden_layers],
            "output_layer": self.output_layer.get_weights(),
        }

    def set_weights(self, state: dict) -> None:
        """Restaura os pesos e biases de todas as camadas a partir de um snapshot."""
        for layer, layer_state in zip(self.hidden_layers, state["hidden_layers"]):
            layer.set_weights(layer_state)
        self.output_layer.set_weights(state["output_layer"])

    def reset_to_initial_weights(self) -> None:
        """Restaura a rede neural para os seus pesos iniciais e zera os deltas de momentum."""
        self.set_weights(self.initial_weights)

    def set_initial_weights(self, weights: dict) -> None:
        """Define e aplica os pesos iniciais da rede a partir de um dicionário de estado."""
        if not isinstance(weights, dict):
            raise TypeError(f"Dicionário de pesos esperado, recebido: {type(weights)}")

        self.set_weights(weights)
        self.initial_weights = self.get_weights()
