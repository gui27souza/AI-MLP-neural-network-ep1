# 4. Conteúdo Direto dos Slides 6 a 13 (Resultados Finais da Fase 3)

## Slide 6 — Atualização de pesos sinápticos
- **No slide:**
  - Código de `update_weights` e `_update_weights` (`03_codigo_slides_2_a_7.md`).
  - Fórmulas: $\Delta w_{jk}(t) = \alpha \delta_k z_j + \mu \Delta w_{jk}(t-1)$ e $\Delta v_{ij}(t) = \alpha \delta_j x_i + \mu \Delta v_{ij}(t-1)$.
- **Falar:**
  - Calcula todos os deltas ($\delta_k$ e $\delta_j$) antes de alterar os pesos (mantém $W$ original no Passo 7).
  - O método `Layer.update_weights` atualiza tanto $V$ quanto $W$ e seus respectivos biases.
  - `momentum = 0.0` executa o Gradiente Descendente puro; `momentum > 0` adiciona o termo inercial.

---

## Slide 7 — Resposta de reconhecimento de caractere
- **No slide:**
  - Código de `predict_class` e `recognize_character` (`03_codigo_slides_2_a_7.md`).
- **Falar:**
  - Saída com 26 neurônios ativados por Sigmoide Bipolar em $(-1, +1)$.
  - Decisão **Winner-Takes-All (`argmax`)**: maior ativação contínua $y_k$ define a letra (`'A'..'Z'`).
  - Exporta saídas contínuas (`saidas_teste_continuas.csv`) e discretas em $\{-1, +1\}$ (`saidas_teste_discretas.csv`).

---

## Slide 8 — Determinação de parâmetros
- **No slide:**
  - Chamada CLI: `make full_pipeline NUM_EPOCHS=120 HIDDEN_NEURONS=48 LEARN_RATE=0.005 MOMENTUM=0.85 PATIENCE=5 VAL_RATIO=0.30`
  - Gráficos do Funil de Busca (`results/fase3_experimentos_finais/1_busca_melhor_modelo/`):
    - `acc83.8_h48_2026_11_17_18_19/grid_search_hyperparams.png` (varredura de $H$, `Val Ratio`, $\eta$ e $\mu$)
    - `acc85.4_h48_2026_11_17_19_07/grid_search_heatmaps.png` (interação $\eta \times \mu$ cravando **85,38%**)
    - `acc85.4_h48_2026_11_17_19_14/grid_search_heatmaps.png` (zoom fino: média de **84,19%** no teste e **87,33%** na validação)
- **Falar:**
  - **Estratégia em Funil (3 fases):** Exploração ampla inicial $\rightarrow$ isolamento das 130 amostras fixas de teste $\rightarrow$ 4 grids progressivos de refinamento até o recorde de **85,38%** (`111/130` acertos).
  - **Sensibilidade:** $H=48$ superou $H=16$ (underfitting: `66,8%` teste) e $H=64$ (`79,7%` média teste, mais lento); `LR=0.005` combinada com `M=0.85` e `PATIENCE=5` evitou sobreajuste e atingiu o pico de generalização.

---

## Slide 9 — Dinâmica de treinamento e teste
- **No slide:**
  - **Hardware:** `Intel Core i5-14600KF (12+8 @ 5.30 GHz) | 32 GB DDR5 6000MT/s Kingston | CPU single-core`
  - **Gráfico de Custo:** `results/fase3_experimentos_finais/1_busca_melhor_modelo/acc83.8_h48_2026_11_17_18_19/grid_search_insights.png` (ou `fase2_conjunto_teste/1_grid_hiperparametros_oficiais/grid_search/grid_search_insights.png`)
  - **Tabela 1 — Neurônios Ocultos ($H$) $\times$ Épocas $\times$ Tempo:**

| Topologia ($H$) | Pesos Treináveis | Épocas Médias | Tempo Médio Total | Custo por Época | Acc Val Média (Máx) | Acc Teste Máx |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$H = 16$** | $2.378$ | $50,3\text{ ep}$ | $15,69\text{ s}$ | **$0,31\text{ s/ep}$** | $76,45\%$ ($79,06\%$) | $70,77\%$ |
| **$H = 48$** | $7.082$ | $43,2\text{ ep}$ | $34,32\text{ s}$ | **$0,79\text{ s/ep}$** | **$87,16\%$ ($88,46\%$)** | **$85,38\%$** |
| **$H = 64$** | $9.434$ | $46,8\text{ ep}$ | $49,27\text{ s}$ | **$1,05\text{ s/ep}$** | $85,51\%$ ($86,81\%$) | $84,62\%$ |

  - **Tabela 2 — Taxa de Aprendizado ($\eta$) e Momentum ($\mu$) $\times$ Épocas $\times$ Tempo (`H=48, P=5`):**

| Variação | Valor | Épocas Médias | Tempo Médio | Custo / Época | Acc Val Média (Máx) | Acc Teste Média (Máx) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Learn Rate ($\eta$)** | $\eta = 0.001$ | $73,3\text{ ep}$ | $58,29\text{ s}$ | $0,79\text{ s/ep}$ | $87,27\%$ ($88,46\%$) | $83,08\%$ ($83,85\%$) |
| **Learn Rate ($\eta$)** | $\eta = 0.005$ | **$29,0\text{ ep}$** | **$22,97\text{ s}$** | $0,79\text{ s/ep}$ | $87,00\%$ ($87,64\%$) | **$83,59\%$ ($85,38\%$)** |
| **Learn Rate ($\eta$)** | $\eta = 0.025$ | $26,8\text{ ep}$ | $24,24\text{ s}$ | $0,91\text{ s/ep}$ | $85,99\%$ ($88,46\%$) | $79,81\%$ ($82,31\%$) |
| **Otimizador ($\mu$)** | $\mu = 0.00$ (GD Puro) | $119\text{ ep}$ | $95,00\text{ s}$ | $0,80\text{ s/ep}$ | **$88,46\%$** | $83,85\%$ |
| **Otimizador ($\mu$)** | $\mu = 0.85$ (Momentum) | **$27\text{ ep}$** | **$21,56\text{ s}$ (-77%)** | $0,80\text{ s/ep}$ | $87,09\%$ | **$85,38\%$** |

- **Falar:**
  - O **custo por época** depende apenas de $H$ (`0,31s` $\rightarrow$ `0,79s` $\rightarrow$ `1,05s`).
  - O **total de épocas** depende de $\eta$ e $\mu$: na configuração campeã (`LR=0.005`), o Momentum reduziu a convergência de **119 épocas (`95,0s`) para 27 épocas (`21,5s`) — 4,4x mais rápido**!

---

## Slide 10 — Treinamento e parada antecipada
- **No slide:**
  - **Divisão Oficial (`CARACTERES COMPLETO` = 1.326 amostras / 51 alfabetos):**
    - **Treino (70% dev):** Primeiras `832 amostras` (32 alfabetos completos, índices `0..831`) — ou `1.196` quando `VAL_RATIO=0.0`.
    - **Validação (30% dev):** Próximas `364 amostras` (14 alfabetos completos, índices `832..1195`).
    - **Teste Fixo:** Últimas `130 amostras` (5 alfabetos completos, índices `1196..1325`).
  - **2 Execuções Comparadas (`H=48, LR=0.005, VAL_RATIO=0.30` — pasta `results/fase3_experimentos_finais/2_comparativo_cenarios_finais/`):**
    1. **Campeão Com Momentum (`M=0.85, P=5`):** Parou na **Época 27** $\rightarrow$ restaurou **Época 22** (`MSE Treino = 0.00239`, `MSE Val = 0.03451`, `Acc Val = 87,09%`) $\rightarrow$ **Teste: `MSE = 0.05241` | `Acc = 85,38%` (`21,56s`)**.
    2. **GD Puro Sem Momentum (`M=0.00, P=10`):** Parou na **Época 119** $\rightarrow$ restaurou **Época 109** (`MSE Treino = 0.00384`, `MSE Val = 0.03513`, `Acc Val = 88,46%`) $\rightarrow$ **Teste: `MSE = 0.05582` | `Acc = 83,85%` (`95,00s`)**.

---

## Slide 11 — Parada antecipada (Gráficos e Ambiente Dificultado)
- **No slide (Curvas de Aprendizado de `results/fase3_experimentos_finais/2_comparativo_cenarios_finais/`):**
  1. **Padrão Estratificado Limpo (`85,38%`):** `1_campeao_oficial_momentum_es_acc85.4/learning_curves.png`
  2. **Ambiente Dificultado 1 — Sem Estratificar (`STRATIFIED=false`, `80,77%`):** `4_sem_estratificacao_aleatorio_acc80.8/learning_curves.png`
  3. **Ambiente Dificultado 2 — Com Ruído 15% (`NOISE_RATE=0.15`, `76,92%`):** `5_com_ruido_15pct_acc76.9/learning_curves.png`
- **Falar:**
  - A linha vertical verde marca o ponto exato de **menor MSE de validação**, e a estrela vermelha (`★`) mostra o erro e a acurácia nos 130 exemplos fixos de teste.
  - **Sem Estratificação (`STRATIFIED=false`):** O sorteio aleatório desbalanceia as classes; o `MSE Val` parece baixo (`0.03386`), mas a acurácia no teste **cai de `85,38%` para `80,77%` (`'-4,61 pp'`)**.
  - **Com Ruído (`15%` de pixels invertidos):** Dificulta a convergência, elevando o `MSE Val` para `0.03664` e reduzindo o teste para **`76,92%` (`'-8,46 pp'`)**.

---

## Slide 12 — Matriz de confusão
- **No slide:**
  - **Matrizes de Confusão (`results/fase3_experimentos_finais/2_comparativo_cenarios_finais/`):**
    - Com Parada Antecipada (`85,38%`): `1_campeao_oficial_momentum_es_acc85.4/confusion_matrix.png`
    - Sem Parada Antecipada — 80 épocas (`80,00%`): `3_sem_parada_antecipada_80ep_acc80.0/confusion_matrix.png`
    - GD Puro (`83,85%`): `2_gd_puro_sem_momentum_es_acc83.8/confusion_matrix.png`
  - **Tabela Comparativa dos 5 Cenários Finais (`H=48, LR=0.005`):**

| Cenário Experimental | Momentum ($\mu$) | Parada Antecipada (Melhor / Total) | Tempo | Acc Treino | Acc Val (MSE Val) | Acc Teste (130 fixos) | MSE Teste |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Campeão Oficial (Estratificado + ES)** | $0.85$ | **Sim (Ép. 22 / 27)** | **$21,6\text{ s}$** | $99,52\%$ | $87,09\%$ ($0.03451$) | **$85,38\%$ (111/130)** | **$0.05241$** |
| **2. GD Puro (Sem Momentum + ES)** | $0.00$ | Sim (Ép. 109 / 119) | $95,0\text{ s}$ | $99,64\%$ | **$88,46\%$** ($0.03513$) | **$83,85\%$ (109/130)** | $0.05582$ |
| **3. Sem Parada Antecipada (80 épocas)** | $0.85$ | **Não (80 / 80)** | $84,2\text{ s}$ | $99,16\%$ | — (`VAL_RATIO=0`) | **$80,00\%$ (104/130)** | $0.05074$ |
| **4. Sem Estratificação (`STRATIFIED=false`)** | $0.85$ | Sim (Ép. 31 / 36) | $29,0\text{ s}$ | $98,81\%$ | $86,35\%$ ($0.03386$) | **$80,77\%$ (105/130)** | $0.05444$ |
| **5. Com Ruído Bipolar (`NOISE_RATE=15%`)** | $0.85$ | Sim (Ép. 30 / 35) | $53,7\text{ s}$ | $98,68\%$ | $84,07\%$ ($0.03664$) | **$76,92\%$ (100/130)** | $0.05753$ |

- **Falar:**
  - **Impacto da Parada Antecipada:** Sem parada antecipada (80 épocas), a rede memoriza o treino (`99,16%`), mas sofre sobreajuste e **perde `5,38 pp` no teste (`85,38%` $\rightarrow$ `80,00%`)**, além de gastar 4x mais tempo.
  - **Análise de Erros nas 130 Amostras:** O campeão acerta 100% (`5/5`) de letras distintas como `Z, Y, V, T, S`, e as raras confusões ocorrem nas fontes estilizadas do final do arquivo em letras morfologicamente próximas na grade $10 \times 12$ (`A/B → R`, `H → M`, `I → J`, `X → K`).

---

## Slide 13 — Implementações adicionais
- **Destaques extras implementados pelo grupo:**
  1. **Hold-out estratificado por alfabeto vs. amostragem aleatória** (`src/data/dataset_loader.py`).
  2. **Otimização com Momentum ($\mu$)** acelerando a convergência em **4,4x** sobre o GD Puro (`src/core/models/layer.py`).
  3. **Rollback automático para o snapshot de menor MSE de validação (`best_weights`)** (`src/training/orchestrator.py`).
  4. **Data Augmentation Bipolar** com inversão estocástica de pixels (`src/data/data_augmentation.py`).
  5. **Validação em problemas binários (`AND, OR, XOR`) e Teste de Mesa automatizado** (`lab/logic_gates/`, `lab/mesa_test/`).
