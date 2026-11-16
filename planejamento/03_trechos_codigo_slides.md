# Trechos de Código e Roteiro de Explicação por Slide (`template-slides.pptx`)

Este guia reúne os recortes exatos do código refatorado para colar nos **Slides 2 a 7** (e no **Slide 10/11**), junto com os pontos-chave que o apresentador deve falar em cada um.

---

## Slide 2 — Estruturas de dados e programação

### 1. Representação da Camada de Entrada (120 pixels) e Saída (26 classes em One-Hot Bipolar)
Arquivo: `src/data/dataset_loader.py` (linhas 48–68)
```python
# 1. Camada de Entrada: 120 pixels bipolares (-1.0 e +1.0) de cada imagem 10x12
with open(filepath_inputs, "r", encoding="utf-8-sig") as f:
    all_inputs = [
        [float(pixel) for pixel in line.strip().split(",")[:120]]
        for line in f if line.strip()
    ]

# 2. Camada de Saída: 26 classes ('A'..'Z') em codificação One-Hot Bipolar
class_labels = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
label_to_index = {label: index for index, label in enumerate(class_labels)}

all_targets: list[list[float]] = []
for label in all_labels:
    target_vector = [-1.0] * len(class_labels)   # -1.0 para classes inativas
    target_vector[label_to_index[label]] = 1.0   # +1.0 apenas na letra correta
    all_targets.append(target_vector)
```

### 2. Camada Escondida, Camada de Saída e Matrizes de Pesos ($V$ e $W$)
Arquivo: `src/core/models/layer.py` (linhas 36–55)
```python
# Biases de cada neurônio da camada (vetor v_0j na oculta ou w_0k na saída)
self.bias: list[float] = list(
    np.random.uniform(low=b_min, high=b_max, size=num_neurons)
)

# Matriz de pesos sinápticos (dimensão: num_neurons x num_inputs)
# - Na camada escondida: matriz V de dimensão (p x 120)
# - Na camada de saída:  matriz W de dimensão (26 x p)
self.weights: list[list[float]] = [
    list(np.random.uniform(low=w_min, high=w_max, size=num_inputs))
    for _ in range(num_neurons)
]
```

---

## Slide 3 — Feedforward e cálculo do erro da rede

### 1. Propagação Direta na Camada (`compute_net_input` e `forward`)
Arquivo: `src/core/models/layer.py` (linhas 62–93)
```python
def compute_net_input(self, i_neuron: int, inputs: list[float]) -> float:
    """z_in_j = v_0j + Σ_i (x_i * v_ij)  [ou y_in_k = w_0k + Σ_j (z_j * w_jk)]"""
    net_in = self.bias[i_neuron]
    for inp, weight in zip(inputs, self.weights[i_neuron]):
        net_in += inp * weight
    return net_in

def forward(self, inputs: list[float]) -> list[float]:
    self.last_inputs = inputs
    outputs: list[float] = []
    for i_neuron in range(self.num_neurons):
        y_in_neuron = self.compute_net_input(i_neuron, inputs)
        outputs.append(self.activation_func(y_in_neuron))
    self.last_outputs = outputs
    return outputs
```

### 2. Feedforward Completo da MLP e Cálculo do Erro Quadrático Médio (MSE)
Arquivo: `src/core/models/mlp.py` (linhas 80–120)
```python
def predict(self, inputs: list[float]) -> list[float]:
    current_activations = inputs
    for layer in self.hidden_layers:
        current_activations = layer.forward(current_activations)
    return self.output_layer.forward(current_activations)

@staticmethod
def compute_mse(outputs_arr: np.ndarray, targets_arr: np.ndarray) -> float:
    """Calcula o Erro Quadrático Médio (MSE) na saída da rede neural."""
    return float(np.mean((targets_arr - outputs_arr) ** 2))
```

---

## Slide 4 — Backpropagation: cálculo do erro para retropropagação (Camada de Saída)

Arquivo: `src/core/models/layer.py` (linhas 99–109) e `src/core/funcs/sigmoide.py` (linhas 29–33)
```python
def deriv_sigmoide_bipolar(y: float) -> float:
    """f'(y_in) = 0.5 * (1 - y) * (1 + y), onde y = f(y_in)"""
    return 0.5 * (1 - y) * (1 + y)

def compute_output_deltas(
    self, target: list[float], outputs: list[float]
) -> list[float]:
    """
    Passo 6 de Fausett — Termo de erro na camada de saída:
    δ_k = (t_k - y_k) * f'(y_in_k)
    """
    output_deltas: list[float] = []
    for t_k, y_k in zip(target, outputs):
        delta_k = (t_k - y_k) * self.derivative_activation(y_k)
        output_deltas.append(delta_k)
    return output_deltas
```

---

## Slide 5 — Backpropagation: contribuição do erro na camada escondida

Arquivo: `src/core/models/layer.py` (linhas 111–133)
```python
def compute_hidden_deltas(
    self,
    next_layer_deltas: list[float],
    next_layer_weights: list[list[float]],
) -> list[float]:
    """
    Passo 7 de Fausett — Contribuição do erro na camada escondida:
    1. Soma ponderada dos erros da saída: δ_in_j = Σ_k (δ_k * w_jk)
    2. Multiplicação pela derivada:       δ_j = δ_in_j * f'(z_in_j)
    """
    hidden_deltas: list[float] = []
    for j_neuron in range(self.num_neurons):
        delta_in_j = 0.0
        for k_next, delta_k in enumerate(next_layer_deltas):
            weight_jk = next_layer_weights[k_next][j_neuron]
            delta_in_j += delta_k * weight_jk

        z_j = self.last_outputs[j_neuron]
        delta_j = delta_in_j * self.derivative_activation(z_j)
        hidden_deltas.append(delta_j)

    return hidden_deltas
```

---

## Slide 6 *(Seu Slide)* — Atualização de pesos sinápticos

### 1. Regra Delta (Gradiente Descendente + Momentum opcional) na Camada
Arquivo: `src/core/models/layer.py` (linhas 139–169)
```python
def update_weights(
    self,
    deltas: list[float],
    learn_rate: float,
    momentum: float = 0.0,
) -> None:
    """
    Passo 8 de Fausett — Atualização de pesos e biases via Gradiente Descendente:
    - Bias:  Δw_0  = α * δ + μ * Δw_0(t-1)        =>  w_0  <- w_0  + Δw_0
    - Pesos: Δw_ji = α * δ * x_i + μ * Δw_ji(t-1) =>  w_ji <- w_ji + Δw_ji
    """
    for i_neuron in range(self.num_neurons):
        delta_i = deltas[i_neuron]

        # Correção e atualização do bias (v_0j ou w_0k)
        delta_bias = (learn_rate * delta_i) + (
            momentum * self.last_delta_bias[i_neuron]
        )
        self.bias[i_neuron] += delta_bias
        self.last_delta_bias[i_neuron] = delta_bias

        # Correção e atualização dos pesos (v_ij ou w_jk)
        for j_input in range(self.num_inputs):
            inp_val = self.last_inputs[j_input]
            delta_w = (learn_rate * delta_i * inp_val) + (
                momentum * self.last_delta_weights[i_neuron][j_input]
            )
            self.weights[i_neuron][j_input] += delta_w
            self.last_delta_weights[i_neuron][j_input] = delta_w
```

### 2. Orquestração dos Passos 6, 7 e 8 na MLP
Arquivo: `src/core/models/mlp.py` (linhas 233–243)
```python
def _update_weights(self, target: list[float], outputs: list[float]) -> None:
    output_deltas = self._compute_output_deltas(target, outputs)
    hidden_deltas_per_layer = self._compute_hidden_deltas(output_deltas)
    self._apply_weight_updates(output_deltas, hidden_deltas_per_layer)
```
> **Dica de fala no vídeo:** Destaque que primeiro calculamos todos os deltas ($\delta_k$ da saída e $\delta_j$ da camada escondida usando os pesos $W$ originais antes da atualização) e em seguida aplicamos `update_weights` em ambas as camadas. Ressalte também que quando `momentum = 0.0` temos o **Gradiente Descendente puro**, e quando `momentum > 0` ativamos a inércia do Momentum!

---

## Slide 7 *(Seu Slide)* — Resposta de reconhecimento de caractere

Arquivo: `src/core/models/mlp.py` (linhas 95–112)
```python
def predict_class(self, outputs: list[float]) -> int:
    """
    Regra Winner-Takes-All (argmax):
    Retorna o índice (0 a 25) do neurônio de saída com maior ativação contínua y_k.
    """
    return int(np.argmax(outputs))

def recognize_character(
    self, inputs: list[float], class_labels: list[str] | None = None
) -> tuple[int, str, list[float]]:
    outputs = self.predict(inputs)
    pred_idx = self.predict_class(outputs)
    labels = class_labels or [chr(ord("A") + i) for i in range(self.num_outputs)]
    pred_char = labels[pred_idx] if pred_idx < len(labels) else str(pred_idx)
    return pred_idx, pred_char, outputs
```
> **Dica de fala no vídeo:** Explique que, embora a rede seja treinada com alvos contínuos em $\{-1.0, +1.0\}$ usando a Sigmoide Bipolar, na fase de inferência os 26 neurônios produzem graus de ativação reais em $(-1, +1)$. O reconhecimento do caractere escolhe o neurônio de maior ativação (`argmax`) e mapeia o índice `0..25` de volta para `'A'..'Z'` (e também gera o vetor discretizado colocando `'+1'` no vencedor e `'-1'` nos demais para o arquivo `saidas_teste_discretas.csv`).

---

## Slide 10 / 11 *(Seus Slides)* — Controle da Parada Antecipada por MSE

Arquivo: `src/training/orchestrator.py` (linhas 94–118 e 178–180)
```python
def _evaluate_val(self, epoch: int) -> tuple[float | None, float | None]:
    if not self.val_inputs or not self.val_targets:
        return None, None

    mse, acc = self.mlp.evaluate(self.val_inputs, self.val_targets)
    self.history["val_mse"].append(mse)
    self.history["val_acc"].append(acc)

    # Monitora a redução do Erro Quadrático Médio (MSE) na validação
    is_better = (mse < self.best_val_mse) or (
        mse == self.best_val_mse and acc > self.best_val_acc
    )
    if is_better:
        self.best_val_mse = mse
        self.best_val_acc = acc
        self.best_epoch = epoch
        self.best_weights = self.mlp.get_weights()  # Snapshot dos melhores pesos
        self._patience_counter = 0
    else:
        self._patience_counter += 1

    return mse, acc
```
E ao encerrar o treinamento (`Orchestrator.run`, linha 179):
```python
# Restaura os pesos da época de menor MSE de validação antes de avaliar o Teste
if self.best_weights is not None:
    self.mlp.set_weights(self.best_weights)
```
