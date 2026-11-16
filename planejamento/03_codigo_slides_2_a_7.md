# 3. Recortes de Código para os Slides 2 a 7

## Slide 2 — Estruturas de dados e programação
- **Entrada (120 pixels) e Saída (26 classes One-Hot Bipolar):** `src/data/dataset_loader.py` (linhas 48–68)
```python
with open(filepath_inputs, "r", encoding="utf-8-sig") as f:
    all_inputs = [
        [float(pixel) for pixel in line.strip().split(",")[:120]]
        for line in f if line.strip()
    ]

class_labels = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
label_to_index = {label: index for index, label in enumerate(class_labels)}

all_targets: list[list[float]] = []
for label in all_labels:
    target_vector = [-1.0] * len(class_labels)
    target_vector[label_to_index[label]] = 1.0
    all_targets.append(target_vector)
```
- **Camada e Pesos ($V$ e $W$):** `src/core/models/layer.py` (linhas 36–55)
```python
self.bias: list[float] = list(
    np.random.uniform(low=b_min, high=b_max, size=num_neurons)
)
self.weights: list[list[float]] = [
    list(np.random.uniform(low=w_min, high=w_max, size=num_inputs))
    for _ in range(num_neurons)
]
```

---

## Slide 3 — Feedforward e cálculo do erro da rede
- **Camada (`compute_net_input` e `forward`):** `src/core/models/layer.py` (linhas 62–93)
```python
def compute_net_input(self, i_neuron: int, inputs: list[float]) -> float:
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
- **MLP (`predict` e `compute_mse`):** `src/core/models/mlp.py` (linhas 80–120)
```python
def predict(self, inputs: list[float]) -> list[float]:
    current_activations = inputs
    for layer in self.hidden_layers:
        current_activations = layer.forward(current_activations)
    return self.output_layer.forward(current_activations)

@staticmethod
def compute_mse(outputs_arr: np.ndarray, targets_arr: np.ndarray) -> float:
    return float(np.mean((targets_arr - outputs_arr) ** 2))
```

---

## Slide 4 — Backpropagation: erro na camada de saída ($\delta_k$)
- `src/core/funcs/sigmoide.py` (linhas 29–33) e `src/core/models/layer.py` (linhas 99–109)
```python
def deriv_sigmoide_bipolar(y: float) -> float:
    return 0.5 * (1 - y) * (1 + y)

def compute_output_deltas(
    self, target: list[float], outputs: list[float]
) -> list[float]:
    output_deltas: list[float] = []
    for t_k, y_k in zip(target, outputs):
        delta_k = (t_k - y_k) * self.derivative_activation(y_k)
        output_deltas.append(delta_k)
    return output_deltas
```

---

## Slide 5 — Backpropagation: contribuição do erro na camada escondida ($\delta_j$)
- `src/core/models/layer.py` (linhas 111–133)
```python
def compute_hidden_deltas(
    self,
    next_layer_deltas: list[float],
    next_layer_weights: list[list[float]],
) -> list[float]:
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

## Slide 6 — Atualização de pesos sinápticos ($\Delta W, \Delta V$)
- `src/core/models/layer.py` (linhas 139–169) e `src/core/models/mlp.py` (linhas 233–243)
```python
def update_weights(
    self, deltas: list[float], learn_rate: float, momentum: float = 0.0
) -> None:
    for i_neuron in range(self.num_neurons):
        delta_i = deltas[i_neuron]

        delta_bias = (learn_rate * delta_i) + (
            momentum * self.last_delta_bias[i_neuron]
        )
        self.bias[i_neuron] += delta_bias
        self.last_delta_bias[i_neuron] = delta_bias

        for j_input in range(self.num_inputs):
            inp_val = self.last_inputs[j_input]
            delta_w = (learn_rate * delta_i * inp_val) + (
                momentum * self.last_delta_weights[i_neuron][j_input]
            )
            self.weights[i_neuron][j_input] += delta_w
            self.last_delta_weights[i_neuron][j_input] = delta_w
```
```python
def _update_weights(self, target: list[float], outputs: list[float]) -> None:
    output_deltas = self._compute_output_deltas(target, outputs)
    hidden_deltas_per_layer = self._compute_hidden_deltas(output_deltas)
    self._apply_weight_updates(output_deltas, hidden_deltas_per_layer)
```

---

## Slide 7 — Resposta de reconhecimento de caractere
- `src/core/models/mlp.py` (linhas 95–112)
```python
def predict_class(self, outputs: list[float]) -> int:
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
