# 4. Estrutura dos Slides 6 a 12

## Slide 6 — Atualização de pesos sinápticos
- **No slide:**
  - Código de `update_weights` e `_update_weights` (ver `03_codigo_slides_2_a_7.md`).
  - Fórmulas: $\Delta w_{jk}(t) = \alpha \delta_k z_j + \mu \Delta w_{jk}(t-1)$ e $\Delta v_{ij}(t) = \alpha \delta_j x_i + \mu \Delta v_{ij}(t-1)$.
- **Pontos da fala:**
  - Calcula todos os deltas ($\delta_k$ e $\delta_j$) antes de alterar os pesos (mantém $W$ original no Passo 7).
  - O método `Layer.update_weights` atualiza tanto $V$ quanto $W$ e seus respectivos biases.
  - `momentum = 0.0` executa o Gradiente Descendente puro; `momentum > 0` adiciona o termo inercial.

---

## Slide 7 — Resposta de reconhecimento de caractere
- **No slide:**
  - Código de `predict_class` e `recognize_character` (ver `03_codigo_slides_2_a_7.md`).
- **Pontos da fala:**
  - Saída com 26 neurônios ativados por Sigmoide Bipolar em $(-1, +1)$.
  - Decisão **Winner-Takes-All (`argmax`)**: maior ativação contínua $y_k$ define a letra (`'A'..'Z'`).
  - Exporta saídas contínuas (`saidas_teste_continuas.csv`) e discretas em $\{-1, +1\}$ (`saidas_teste_discretas.csv`).

---

## Slide 8 — Determinação de parâmetros *(Em andamento — aguardando novos testes)*
- **O que o slide pede:**
  - Como os parâmetros são definidos no código/CLI (`make full_pipeline ...`).
  - Quais parâmetros podem ser variados (`HIDDEN_NEURONS`, `LEARN_RATE`, `MOMENTUM`, `VAL_RATIO`, `PATIENCE`, `NOISE_RATE`, `STRATIFIED`).
  - Gráficos/tabelas da busca de parâmetros e análise de quais parâmetros mais influenciam.

---

## Slide 9 — Dinâmica de treinamento e teste *(Em andamento — aguardando novos testes)*
- **Hardware:** `Intel Core i5-14600KF (12+8 @ 5.30 GHz) | 32 GB DDR5 6000MT/s Kingston`
- **O que o slide pede:**
  - Tempo gasto variando:
    1. Neurônios ocultos ($H$) $\times$ tempo $\times$ épocas
    2. Taxa de aprendizado ($\eta$) $\times$ tempo $\times$ épocas
    3. Quantidade de épocas $\times$ tempo

---

## Slide 10 — Treinamento e parada antecipada *(Em andamento — aguardando novos testes)*
- **Divisão Oficial do Dataset (`CARACTERES COMPLETO` = 1.326 amostras / 51 alfabetos):**
  - **Treinamento:** Primeiras `962 amostras` (37 alfabetos completos, com `VAL_RATIO=0.20`) — ou `1.196 amostras` sem validação (`VAL_RATIO=0.0`).
  - **Validação:** `234 amostras` (9 alfabetos completos, `VAL_RATIO=0.20`).
  - **Teste Fixo:** Últimas `130 amostras` (5 alfabetos completos, índices `1196..1325`).
- **O que o slide pede:**
  - Pelo menos 2 execuções no terminal destacando a queda do MSE no treino e validação, erro final no teste, parametrização e convergência dos pesos.

---

## Slide 11 — Parada antecipada — Gráficos *(Em andamento — aguardando novos testes)*
- **O que o slide pede:**
  - Gráficos das curvas de aprendizado (MSE de treino vs. validação com marcação da melhor época).
  - Comparação dificultando o ambiente: amostragem aleatória sem estratificação (`STRATIFIED=false`) e/ou ruído (`NOISE_RATE`).

---

## Slide 12 — Matriz de confusão *(Em andamento — aguardando novos testes)*
- **O que o slide pede:**
  - Plotagem de matrizes de confusão variando as condições (variação de parâmetros, treinamento completo sem parada antecipada, uso de parada antecipada).
  - Análise crítica dos acertos e das principais confusões no conjunto de teste (130 amostras).

---

## Slide 13 — Implementações adicionais
- **O que apresentar (caso sobre tempo dentro dos 20 min):**
  - **Hold-out estratificado vs. aleatório** (`src/data/dataset_loader.py`).
  - **Otimização com Momentum ($\mu$)** além do Gradiente Descendente puro (`src/core/models/layer.py`).
  - **Data Augmentation Bipolar** com inversão estocástica de pixels (`src/data/data_augmentation.py`).
  - **Rollback automático para os melhores pesos de validação (`best_weights`)** (`src/training/orchestrator.py`).
  - **Validação em problemas binários (Portas Lógicas `AND, OR, XOR`) e Teste de Mesa** (`lab/logic_gates/`, `lab/mesa_test/`).
