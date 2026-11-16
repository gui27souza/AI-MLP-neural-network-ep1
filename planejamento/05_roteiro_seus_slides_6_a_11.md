# Roteiro Detalhado dos Seus Slides (Slides 6 a 11)

Este documento organiza exatamente o que colocar e o que falar nos **seus 6 slides** (`Slide 6` ao `Slide 11` do `template-slides.pptx`), garantindo que cada bullet exigido pela professora seja coberto com precisão.

---

## Slide 6 — Atualização de pesos sinápticos
- **O que colocar no slide:**
  - Recorte do método `Layer.update_weights` (`src/core/models/layer.py`, linhas 139–169) e `MLP._update_weights` (`src/core/models/mlp.py`, linhas 233–243) — disponíveis em `03_trechos_codigo_slides.md`.
  - As equações da Regra Delta:
    - $\Delta w_{jk}(t) = \alpha \cdot \delta_k \cdot z_j + \mu \cdot \Delta w_{jk}(t-1) \implies w_{jk} \leftarrow w_{jk} + \Delta w_{jk}(t)$
    - $\Delta v_{ij}(t) = \alpha \cdot \delta_j \cdot x_i + \mu \cdot \Delta v_{ij}(t-1) \implies v_{ij} \leftarrow v_{ij} + \Delta v_{ij}(t)$
- **Pontos-chave para falar:**
  1. Os deltas ($\delta_k$ da saída e $\delta_j$ da camada oculta) são todos calculados **antes** de alterar qualquer peso, garantindo que a retropropagação use a matriz $W$ da iteração atual (exatamente como nos Passos 6, 7 e 8 de Fausett).
  2. Como tanto a camada oculta quanto a camada de saída são instâncias da classe `Layer`, o método `update_weights` é reaproveitado para atualizar ambas as matrizes ($W$ e $V$) e seus respectivos vetores de bias ($w_{0k}$ e $v_{0j}$).
  3. Quando `momentum = 0.0`, temos o **Gradiente Descendente puro** (requisito obrigatório); quando `momentum > 0.0`, ativamos o termo inercial que soma uma fração $\mu$ da atualização da amostra anterior (`last_delta_weights`).

---

## Slide 7 — Resposta de reconhecimento de caractere
- **O que colocar no slide:**
  - Recorte dos métodos `MLP.predict_class` e `MLP.recognize_character` (`src/core/models/mlp.py`, linhas 95–112).
- **Pontos-chave para falar:**
  1. A camada de saída possui $m = 26$ neurônios (um para cada letra de `'A'` a `'Z'`), ativados pela função **Sigmoide Bipolar** no intervalo contínuo $(-1, +1)$.
  2. Durante o treinamento, o vetor alvo $t$ usa codificação *One-Hot Bipolar*: $+1.0$ no índice da letra verdadeira e $-1.0$ nas outras 25 posições.
  3. Para reconhecer o caractere na inferência, aplicamos a estratégia **Winner-Takes-All (`argmax`)**: identificamos o neurônio $k \in \{0..25\}$ que produziu a maior ativação contínua $y_k$ e convertemos seu índice na letra correspondente (`chr(ord('A') + pred_idx)`).
  4. Para fins de entrega (`saidas_teste_continuas.csv` e `saidas_teste_discretas.csv`), salvamos tanto o vetor bruto contínuo $y \in (-1, +1)^{26}$ quanto o vetor discretizado em $\{-1, +1\}^{26}$.

---

## Slide 8 — Determinação de parâmetros
- **O que a professora pede neste slide:**
  1. *Como você determina os parâmetros da arquitetura (no código, scripts, interface)?*
  2. *Quais são os parâmetros que podemos variar quando estivermos executando-a?*
  3. *Como você fez o procedimento de busca pelos melhores valores (use gráficos e tabelas) e quais parâmetros influenciam mais ou menos?*
- **O que colocar no slide:**
  - **Configuração via CLI / Scripts:** Mostrar um exemplo curto de chamada `make full_pipeline NUM_EPOCHS=... HIDDEN_NEURONS=...` e a tabela de variáveis parametrizáveis (`NUM_EPOCHS`, `HIDDEN_NEURONS`, `LEARN_RATE`, `MOMENTUM`, `VAL_RATIO`, `PATIENCE`, `TARGET_MSE`, `NOISE_RATE`, `STRATIFIED`, `SEED`).
  - **Gráficos de Busca em Grade (Grid Search):**
    - Usar o painel `grid_search_hyperparams.png` e `grid_search_heatmaps.png` (tanto da exploração ampla inicial quanto do Grid Search 1 refinado).
- **Pontos-chave para falar (Análise de Sensibilidade):**
  1. **Neurônios na Camada Escondida ($H$):** É o fator estrutural mais determinante até atingir a capacidade suficiente — com $H=16$ a rede sofre *underfitting* (capacidade insuficiente para separar as 26 classes), enquanto em **$H=48$** atinge o ponto ótimo de generalização. Acima disso ($H=64, 96, 144$), o ganho de acurácia estagna e o custo computacional cresce linearmente.
  2. **Taxa de Aprendizado ($\eta$) e Momentum ($\mu$):** Possuem forte interação acoplada — taxas moderadas ($\eta \in [0.015, 0.025]$) combinadas com momentum intermediário ($\mu = 0.55$) aceleram a queda do MSE em menos épocas. Já a combinação de $\eta$ alto com $\mu$ muito alto ($\mu \ge 0.85$) desestabiliza o gradiente online.
  3. **Proporção de Validação (`VAL_RATIO`):** Reservar `20%` (9 alfabetos = 234 amostras) da base de desenvolvimento equilibra um conjunto de validação representativo sem sacrificar amostras de treinamento (`962` amostras = 37 alfabetos).

---

## Slide 9 — Dinâmica de treinamento e teste
- **O que a professora pede neste slide:**
  1. *Hardware utilizado para executar os experimentos.*
  2. *Tempo gasto para executar treinamentos variando: (a) neurônios ocultos $\times$ tempo $\times$ épocas; (b) taxa de aprendizado $\times$ tempo $\times$ épocas; (c) quantidade de épocas $\times$ tempo.*
- **O que colocar no slide:**
  - **Box de Hardware:**
    - **Processador:** Apple M1 Pro (ARM64, 8 núcleos de CPU)
    - **Memória RAM:** 16 GB Unificada
    - **Ambiente:** macOS / Python 3.13.18 (execução 100% em CPU single-core no loop online da MLP)
  - **Tabelas extraídas do `ANALISE_GRID_SEARCH.md` / `summary.csv` (Grid Search 1):**
    - Tabela 1: **Neurônios Ocultos ($H \in \{16, 48, 64\}$)** $\times$ Épocas Executadas $\times$ Tempo Total (s) $\times$ Tempo por Época (s/época) $\times$ MSE/Acc Teste.
    - Tabela 2: **Taxa de Aprendizado ($\eta \in \{0.008, 0.015, 0.025\}$)** $\times$ Épocas Executadas $\times$ Tempo Total (s) $\times$ MSE/Acc Teste.
    - Gráfico de Custo Computacional (`grid_search_insights.png` — painel direito).
- **Pontos-chave para falar:**
  1. O tempo **por época** depende diretamente de $H$ (complexidade $O(N \cdot (n \cdot p + p \cdot m))$): para $H=16$ é ~0,3s/época, para $H=48$ é ~0,9s/época e para $H=64$ é ~1,2s/época.
  2. Já o **número total de épocas** até a convergência (ou acionamento do Early Stopping) depende da taxa de aprendizado $\eta$ e do momentum $\mu$: taxas muito baixas ($\eta = 0.008$) demoram muito mais épocas para reduzir o MSE, aumentando o tempo total de treinamento.

---

## Slide 10 — Treinamento e parada antecipada
- **O que a professora pede neste slide:**
  1. *Apresentar pelo menos duas execuções no terminal.*
  2. *Destacar a convergência do ERRO (MSE) no treino e na validação, e ao final mostrar o erro no conjunto de teste.*
  3. *Informar: parametrização da rede, convergência dos pesos, qual é o conjunto de treino, qual é o de validação e qual é o de teste.*
- **O que colocar no slide:**
  - **Box claro da Divisão Oficial dos Dados (`CARACTERES COMPLETO` = 1.326 amostras / 51 alfabetos):**
    - **Treinamento:** Primeiros `37 alfabetos` = **962 amostras** (índices `0..961`, 37 exemplos de cada letra A–Z).
    - **Validação:** Próximos `9 alfabetos` = **234 amostras** (índices `962..1195`, 9 exemplos de cada letra A–Z, `VAL_RATIO=0.20`).
    - **Teste (Fixo):** Últimos `5 alfabetos` = **130 amostras** (índices `1196..1325`, 5 exemplos de cada letra A–Z, conforme `README.md`).
  - **Print/Gravação do Terminal das Execuções 4A e 4B:**
    - **Execução 1 (4A):** $H=48, \eta=0.025, \mu=0.0$ (Gradiente Descendente Puro) + `PATIENCE=10`.
    - **Execução 2 (4B):** $H=48, \eta=0.025, \mu=0.55$ (Com Momentum) + `PATIENCE=10`.
    - *(O nosso `Orchestrator.print_results()` já imprime todas as informações pedidas no final do terminal!).*

---

## Slide 11 — Parada antecipada (Gráficos e Ambiente Dificultado)
- **O que a professora pede neste slide:**
  1. *Gráficos ilustrando as execuções de parada antecipada (minimizando o MSE).*
  2. *Se convergir sem grandes variações, dificultar o ambiente: incluir ruído, juntar classes ou fazer amostragem aleatória sem estratificação.*
- **O que colocar no slide:**
  - Gráfico 1: `learning_curves.png` da execução padrão limpa (**4B** — mostrando a linha vertical verde exatamente na época de **menor MSE de validação**).
  - Gráfico 2: `learning_curves.png` da execução **4D (`STRATIFIED=false` — Amostragem Aleatória sem Estratificação)** e/ou com **Ruído (`NOISE_RATE=0.15` do Grid Search 3)**, mostrando a oscilação e o descolamento mais acentuado entre a curva de MSE de treino e a curva de MSE de validação!

---

## Comandos para Rodar no Desktop (Bateria Oficial Fase 2)

Ao fazer `git pull` na branch `ajustes-entrega` no seu Desktop, execute o bloco abaixo para gerar os 3 Grid Searches e as 4 avaliações completas com os tempos reais do hardware do Desktop:

```bash
# 1. Grid Search Principal: Topologia (H) x Learn Rate x Momentum (incluindo 0.0 = GD Puro)
make full_grid_pipeline \
  MAX_EPOCHS=100 \
  VAL_RATIOS="0.20" \
  HIDDEN_NEURONS="16,48,64" \
  LEARN_RATE="0.008,0.015,0.025" \
  MOMENTUM="0.0,0.55,0.85" \
  PATIENCE="10" \
  NOISE_RATES="0.0" \
  TARGET_MSE=0.001 \
  MIN_ACC_THRESHOLD=70.0 \
  SEED=42 && \
sleep 65 && \

# 2. Grid Search: Proporção de Validação (10%, 20%, 30% dos 1.196 de desenvolvimento)
make full_grid_pipeline \
  MAX_EPOCHS=100 \
  VAL_RATIOS="0.10,0.20,0.30" \
  HIDDEN_NEURONS="48" \
  LEARN_RATE="0.015,0.025" \
  MOMENTUM="0.0,0.55" \
  PATIENCE="10" \
  NOISE_RATES="0.0" \
  TARGET_MSE=0.001 \
  MIN_ACC_THRESHOLD=70.0 \
  SEED=42 && \
sleep 65 && \

# 3. Grid Search: Impacto de Ruído / Data Augmentation Bipolar
make full_grid_pipeline \
  MAX_EPOCHS=100 \
  VAL_RATIOS="0.20" \
  HIDDEN_NEURONS="48" \
  LEARN_RATE="0.025" \
  MOMENTUM="0.0,0.55" \
  PATIENCE="10" \
  NOISE_RATES="0.0,0.05,0.15,0.30" \
  AUGMENT_COPIES=1 \
  TARGET_MSE=0.001 \
  MIN_ACC_THRESHOLD=60.0 \
  SEED=42 && \

# 4A. Pipeline Completo: Campeão com Gradiente Descendente Puro (MOMENTUM=0.0) + Early Stopping
make full_pipeline \
  NUM_EPOCHS=100 HIDDEN_NEURONS=48 LEARN_RATE=0.025 MOMENTUM=0.0 PATIENCE=10 VAL_RATIO=0.20 TARGET_MSE=0.001 SEED=42 PRINT_INTERVAL=5 && \

# 4B. Pipeline Completo: Campeão com Momentum (MOMENTUM=0.55) + Early Stopping
make full_pipeline \
  NUM_EPOCHS=100 HIDDEN_NEURONS=48 LEARN_RATE=0.025 MOMENTUM=0.55 PATIENCE=10 VAL_RATIO=0.20 TARGET_MSE=0.001 SEED=42 PRINT_INTERVAL=5 && \

# 4C. Pipeline Completo: Treino SEM Parada Antecipada (Slide 12 - 1.196 treino / 130 teste)
make full_pipeline \
  NUM_EPOCHS=60 HIDDEN_NEURONS=48 LEARN_RATE=0.025 MOMENTUM=0.55 VAL_RATIO=0.0 SEED=42 PRINT_INTERVAL=5 && \

# 4D. Pipeline Completo: Amostragem Aleatória SEM Estratificação (Slide 11)
make full_pipeline \
  NUM_EPOCHS=100 HIDDEN_NEURONS=48 LEARN_RATE=0.025 MOMENTUM=0.55 PATIENCE=10 VAL_RATIO=0.20 STRATIFIED=false TARGET_MSE=0.001 SEED=42 PRINT_INTERVAL=5
```
