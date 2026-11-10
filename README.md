# AI-MLP-neural-network-ep1

Implementação de uma Rede Neural Multicamadas (MLP - Multi-Layer Perceptron) baseada no algoritmo Backpropagation de Laurene Fausett, com suporte a Momentum, Early Stopping e Erro Alvo (Target MSE) para reconhecimento e classificação de caracteres.

---

## Como Começar

### Pré-requisitos e Instalação

Clone o repositório e configure o ambiente virtual Python:

```bash
cd AI-MLP-neural-network-ep1/

# Criação e ativação do ambiente virtual
python3 -m venv venv
source venv/bin/activate

# Instalação das dependências
pip install -r requirements.txt
```

---

## Como Executar

O projeto utiliza um `Makefile` para gerenciar a execução e configuração dos hiperparâmetros.

### 1. Treinamento Principal (Reconhecimento de Caracteres)

Executa o treinamento com a configuração padrão:

```bash
make main
```

#### Customizando Hiperparâmetros via Linha de Comando:

Você pode sobrescrever qualquer variável diretamente na chamada do make:

```bash
# Exemplo 1: Treinar com 50 épocas, 64 neurônios e momentum 0.7
make main NUM_EPOCHS=50 HIDDEN_NEURONS=64 MOMENTUM=0.7

# Exemplo 2: Usar Early Stopping com paciência de 5 épocas
make main NUM_EPOCHS=40 PATIENCE=5

# Exemplo 3: Parar automaticamente quando o MSE atingir 0.01
make main NUM_EPOCHS=30 TARGET_MSE=0.01

# Exemplo 4: Alterar a taxa de aprendizado
make main NUM_EPOCHS=30 LEARN_RATE=0.01
```

| Parâmetro | Padrão | Descrição |
| :--- | :--- | :--- |
| `NUM_EPOCHS` | `20` | Quantidade máxima de épocas de treinamento |
| `LEARN_RATE` | `0.015` | Taxa de aprendizado da rede |
| `HIDDEN_NEURONS` | `32` | Quantidade de neurônios na camada escondida |
| `MOMENTUM` | `0.5` | Fator de momentum para aceleração do gradiente |
| `PATIENCE` | *(vazio)* | Número de épocas sem melhora na validação para interromper o treino |
| `TARGET_MSE` | *(vazio)* | Erro Quadrático Médio alvo para parada antecipada |
| `PRINT_INTERVAL` | `2` | Intervalo de épocas para exibição do progresso no terminal |

---

### 2. Laboratório de Portas Lógicas

Para testar a convergência do algoritmo em problemas lineares e não-lineares simples (AND, OR, XOR):

```bash
make lab_logic_gates NUM_EPOCHS=500
```

---

## Modelos e Checkpoints

Ao final do treinamento (ou em caso de interrupção com `Ctrl+C`), a rede restaura os melhores pesos observados na validação e salva o modelo na pasta `models/` no formato JSON:

```text
models/mlp_caracteres_h32_e14_acc78.5_mse0.057.json
```

O arquivo contém:
- Arquitetura e hiperparâmetros utilizados
- Pesos e biases de todas as camadas
- Histórico e métricas de treino e validação (MSE, acurácia e época de melhor resultado)
