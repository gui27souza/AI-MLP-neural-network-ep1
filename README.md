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

# Exemplo 5: Alterar a proporção do conjunto de validação para 20%
make main NUM_EPOCHS=40 VAL_RATIO=0.2
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
| `VAL_RATIO` | `0.1` | Proporção do dataset reservada para validação/teste (ex: 0.1 = 10%, 0.2 = 20%) |
| `SEED` | `42` | Semente pseudoaleatória para inicialização de pesos e embaralhamento |
| `AUGMENT_DATA` | `0` | Ativa Data Augmentation Bipolar no treino (`1` para ativar) |
| `NOISE_RATE` | `0.05` | Taxa de inversão aleatória de pixels no treino (ex: 0.05 = ~5%) |
| `AUGMENT_COPIES` | `1` | Quantidade de cópias ruidosas geradas por amostra original de treino |

---

### 2. Laboratório de Portas Lógicas

Para testar a convergência do algoritmo em problemas lineares e não-lineares simples (AND, OR, XOR):

```bash
make lab_logic_gates NUM_EPOCHS=500
```

---

### 3. Pipeline Completo (Treino + Avaliação Oficial)

Para executar o treinamento e imediatamente gerar todos os entregáveis oficiais (relatório textual, matriz de confusão, curvas de aprendizado e arquivos formatados do EP1 em `results/`):

```bash
make full_pipeline NUM_EPOCHS=120 HIDDEN_NEURONS=48 LEARN_RATE=0.01 MOMENTUM=0.55 PATIENCE=6 VAL_RATIO=0.3
```

Para treinar com aumento de dados bipolar (5% de inversão de ruído):

```bash
make full_pipeline NUM_EPOCHS=120 HIDDEN_NEURONS=48 LEARN_RATE=0.01 MOMENTUM=0.50 PATIENCE=6 VAL_RATIO=0.3 AUGMENT_DATA=1 NOISE_RATE=0.05
```

---

### 4. Busca em Grade (Grid Search)

Para executar uma busca automatizada sobre combinações de hiperparâmetros com relatório consolidado em CSV e ranking impresso no terminal:

```bash
make grid_search
```

Os resultados são salvos em `models/grid_search/<TIMESTAMP>/summary.csv` juntamente com os arquivos JSON de cada modelo testado na rodada.

---

## Modelos e Checkpoints

Ao final do treinamento (ou em caso de interrupção com `Ctrl+C`), a rede restaura os melhores pesos observados na validação (priorizando a maior acurácia de validação e usando menor MSE como critério de desempate) e salva o modelo na pasta `models/` no formato JSON:

```text
models/mlp_caracteres_acc78.5_h32.json
```

O arquivo contém:
- Arquitetura e hiperparâmetros utilizados
- Pesos e biases de todas as camadas
- Histórico e métricas de treino e validação (MSE, acurácia e época de melhor resultado)
