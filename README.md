# AI-MLP-neural-network-ep1

> **Integrantes do Grupo (EACH-USP | IA 2026 — Entrega 1: MLP):**
> - **Guilherme Oliveira de Souza** — Nº USP: `14671388`
> - **Paulo Rogerio Andrade de Oliveira** — Nº USP: `9365783`
> - **Victor Hugo Antunes Valadares** — Nº USP: `14760317`


Implementação de uma Rede Neural Multicamadas (MLP - Multi-Layer Perceptron) baseada no algoritmo Backpropagation com Gradiente Descendente de Laurene Fausett, com suporte a Momentum, Parada Antecipada (Early Stopping guiado por MSE) e Erro Alvo (Target MSE) para reconhecimento e classificação de caracteres (`CARACTERES COMPLETO`).

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

## Divisão Oficial do Conjunto de Dados (`CARACTERES COMPLETO`)

O dataset possui **1.326 caracteres** ($51$ alfabetos completos de $26$ letras `'A'..'Z'`, matrizes $10 \times 12 = 120$ pixels bipolares em $\{-1, +1\}$):
1. **Conjunto de Teste (Fixo):** Os **últimos 130 exemplos** (índices `1196..1325`, equivalentes aos últimos 5 alfabetos completos) são isolados exclusivamente para avaliação final de teste e geração da Matriz de Confusão.
2. **Base de Desenvolvimento (Primeiros 1.196 exemplos = 46 alfabetos):** Particionada entre **Treinamento** e **Validação** segundo `VAL_RATIO` (alinhada em múltiplos de 26 letras com `STRATIFIED=true` por padrão, ou sorteada sem estratificação com `STRATIFIED=false`). Quando `VAL_RATIO=0.0`, todas as 1.196 amostras são usadas no treinamento (treinamento completo sem parada antecipada).

---

## Como Executar

O projeto utiliza um `Makefile` e scripts `.sh` (`./run.sh` e `./run_grid.sh`) para gerenciar a execução e configuração dos hiperparâmetros.

### 1. Teste de Mesa Oficial e Laboratório de Portas Lógicas

Para validar a corretude matemática passo a passo contra o documento oficial da disciplina (`docs/exemplo-numerico-MLP.pdf`):

```bash
make lab_mesa_test
```

Para testar a convergência do algoritmo em problemas lineares e não-lineares clássicos (`AND`, `OR`, `XOR`):

```bash
make lab_logic_gates NUM_EPOCHS=500
```

---

### 2. Treinamento Principal e Pipeline Completo (Treino + Avaliação Oficial)

Para executar o treinamento e imediatamente gerar todos os entregáveis oficiais (relatório textual, matriz de confusão, curvas de aprendizado e arquivos formatados do EP1 em `results/`):

```bash
./run.sh
# ou via Makefile:
make full_pipeline NUM_EPOCHS=120 HIDDEN_NEURONS=48 LEARN_RATE=0.025 MOMENTUM=0.55 PATIENCE=10 VAL_RATIO=0.2
```

#### Customizando Hiperparâmetros via Linha de Comando:

```bash
# Exemplo 1: Treinamento com Gradiente Descendente Puro (sem Momentum) e Parada Antecipada
make full_pipeline NUM_EPOCHS=120 HIDDEN_NEURONS=48 LEARN_RATE=0.025 MOMENTUM=0.0 PATIENCE=10 VAL_RATIO=0.2

# Exemplo 2: Treinamento Completo SEM Parada Antecipada (1.196 treino / 130 teste)
make full_pipeline NUM_EPOCHS=60 HIDDEN_NEURONS=48 LEARN_RATE=0.025 MOMENTUM=0.55 VAL_RATIO=0.0

# Exemplo 3: Parada Antecipada com amostragem aleatória sem estratificação (Slide 11)
make full_pipeline NUM_EPOCHS=120 HIDDEN_NEURONS=48 LEARN_RATE=0.025 MOMENTUM=0.55 PATIENCE=10 VAL_RATIO=0.2 STRATIFIED=false

# Exemplo 4: Treinamento com Data Augmentation Bipolar (5% de inversão de pixels)
make full_pipeline NUM_EPOCHS=120 HIDDEN_NEURONS=48 LEARN_RATE=0.025 MOMENTUM=0.55 PATIENCE=10 VAL_RATIO=0.2 NOISE_RATE=0.05
```

| Parâmetro | Descrição |
| :--- | :--- |
| `NUM_EPOCHS` | Quantidade máxima de épocas de treinamento |
| `LEARN_RATE` | Taxa de aprendizado da rede ($\alpha$ / $\eta$) |
| `HIDDEN_NEURONS` | Quantidade de neurônios na camada escondida ($p$) |
| `MOMENTUM` | Fator de momentum ($\mu$) para aceleração do gradiente (`0.0` = GD puro) |
| `PATIENCE` | Número de épocas sem redução no **MSE de validação** para interromper o treino |
| `TARGET_MSE` | Erro Quadrático Médio de treino alvo para parada antecipada |
| `PRINT_INTERVAL` | Intervalo de épocas para exibição do progresso no terminal |
| `VAL_RATIO` | Proporção das 1.196 amostras de desenvolvimento reservada para validação (`0.0` desativa validação) |
| `STRATIFIED` | `true` (padrão) fatia em blocos completos de alfabeto (26 letras); `false` amostra aleatoriamente sem estratificação |
| `SEED` | Semente pseudoaleatória para inicialização de pesos e embaralhamento (`42`) |
| `NOISE_RATE` | Taxa de inversão aleatória de pixels no treino (`0.0` = limpo) |
| `AUGMENT_COPIES` | Quantidade de cópias ruidosas geradas por amostra original de treino |

---

### 3. Busca em Grade (Grid Search + Análise Gráfica)

Para executar uma busca automatizada sobre combinações de hiperparâmetros com relatório consolidado (`summary.csv`, `ANALISE_GRID_SEARCH.md` e gráficos comparativos):

```bash
./run_grid.sh
# ou via Makefile:
make full_grid_pipeline MAX_EPOCHS=120 VAL_RATIOS=0.2,0.3 HIDDEN_NEURONS=16,48,64 LEARN_RATE=0.005,0.015,0.025 MOMENTUM=0.0,0.55,0.85 PATIENCE=10
```

---

## Modelos, Checkpoints e Entregáveis Oficiais

Ao final do treinamento, a rede restaura os pesos da época de **menor MSE de validação**, avalia os **130 exemplos fixos de teste**, salva o checkpoint JSON em `models/` e exporta em `results/<MODELO>/saidas_oficiais/`:
- `hiperparametros.txt`: Arquitetura, inicialização, divisão Treino/Validação/Teste e métricas alcançadas.
- `pesos_iniciais.csv` (`+ _V.csv, _W.csv`): Matrizes iniciais no padrão de Fausett (coluna 0 = bias).
- `pesos_finais.csv` (`+ _V.csv, _W.csv`): Matrizes finais treinadas no padrão de Fausett.
- `historico_erro_treinamento.csv`: Evolução de MSE e acurácia por época.
- `saidas_teste_continuas.csv` e `saidas_teste_discretas.csv`: Saídas produzidas para cada um dos 130 dados de teste.
