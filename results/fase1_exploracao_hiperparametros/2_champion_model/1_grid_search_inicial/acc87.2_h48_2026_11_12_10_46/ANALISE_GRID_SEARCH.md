# Relatorio Analitico - Grid Search

Total de Modelos Treinados: 27


## 1. Melhor Modelo da Execucao (Top 1)

- **Rodada:** #17
- **Topologia:** `H = 48`
- **Hiperparametros:** `Val Ratio = 0.3`, `LR = 0.025`, `Momentum = 0.55`, `Patience = 6`, `Noise = 0.0`
- **Melhor Acuracia de Validacao:** **87.15%** (alcancada na epoca 14)
- **Acuracia de Treino Final:** 99.68% (Gap Treino-Val: **+12.53%**)
- **Menor MSE de Validacao:** **0.03918**
- **Epocas Executadas:** 20 (20.28s) | Parada Antecipada: `Sim`
- **Checkpoint Salvo:** `gs_acc87.2_h48.json`


## 2. Comparativo por Topologia (Neuronios Ocultos H)

| Neuronios (H) | Total Modelos | Acuracia Media | Acuracia Maxima | Gap Overfit Medio | MSE Minimo | Tempo Medio (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H=16** | 9 | 71.12% | **75.57%** | +23.09% | **0.06109** | 14.14s |
| **H=48** | 9 | 85.05% | **87.15%** | +14.13% | **0.03865** | 26.41s |
| **H=64** | 9 | 85.08% | **86.40%** | +14.27% | **0.04029** | 42.83s |

## 3. Comparativo por Taxa de Aprendizado (Learn Rate)

| Taxa (η) | Total Modelos | Acuracia Media | Acuracia Maxima | Gap Overfit Medio | MSE Minimo | Tempo Medio (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **η = 0.008** | 9 | 81.30% | **86.40%** | +17.07% | **0.03865** | 33.28s |
| **η = 0.01** | 9 | 81.36% | **86.40%** | +17.03% | **0.03876** | 25.75s |
| **η = 0.025** | 9 | 78.59% | **87.15%** | +17.40% | **0.03918** | 24.35s |

## 4. Comparativo por Fator de Momentum

| Momentum (α) | Total Modelos | Acuracia Media | Acuracia Maxima | Gap Overfit Medio | MSE Minimo | Tempo Medio (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **α = 0.15** | 9 | 81.25% | **85.89%** | +17.02% | **0.04142** | 32.22s |
| **α = 0.55** | 9 | 82.03% | **87.15%** | +16.69% | **0.03918** | 27.73s |
| **α = 0.85** | 9 | 77.97% | **86.40%** | +17.79% | **0.03865** | 23.43s |

## 5. Diagnostico de Generalizacao & Overfitting

- **Gap Medio Global (Treino - Validacao):** **+17.17%**
- **Modelo Mais Generalizador (Menor Gap):** Rodada #17 (Treino: 99.68% | Val: 87.15% | Gap: **+12.53%**)
- **Distribuicao da Capacidade de Generalizacao:**
  - Generalizacao Alta (Gap <= 5%): **0/27** modelos
  - Sobreajuste Moderado (5% < Gap <= 15%): **16/27** modelos
  - Sobreajuste Acentuado (Gap > 15%): **11/27** modelos


## 6. Top 10 Melhores Modelos da Execucao

| Pos | Rodada | H | Ratio | LR | Momentum | P | Noise | Acc Treino | Acc Val | Gap Overfit | MSE Val | Epocas | Tempo | Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| #1 | #17 | 48 | 0.30 | 0.025 | 0.55 | 6 | 0.00 | 99.68% | **87.15%** | +12.53% | 0.03918 | 20 | 20.3s | `gs_acc87.2_h48.json` |
| #2 | #12 | 48 | 0.30 | 0.008 | 0.85 | 6 | 0.00 | 99.68% | **86.40%** | +13.28% | 0.03865 | 29 | 30.4s | `gs_acc86.4_h48.json` |
| #3 | #15 | 48 | 0.30 | 0.01 | 0.85 | 6 | 0.00 | 99.68% | **86.40%** | +13.28% | 0.03876 | 26 | 26.9s | `gs_acc86.4_h48_v2.json` |
| #4 | #26 | 64 | 0.30 | 0.025 | 0.55 | 6 | 0.00 | 99.57% | **86.40%** | +13.17% | 0.04029 | 35 | 45.5s | `gs_acc86.4_h64.json` |
| #5 | #21 | 64 | 0.30 | 0.008 | 0.85 | 6 | 0.00 | 99.57% | **86.15%** | +13.42% | 0.04067 | 30 | 39.3s | `gs_acc86.1_h64.json` |
| #6 | #16 | 48 | 0.30 | 0.025 | 0.15 | 6 | 0.00 | 99.57% | **85.89%** | +13.68% | 0.04142 | 30 | 32.2s | `gs_acc85.9_h48.json` |
| #7 | #23 | 64 | 0.30 | 0.01 | 0.55 | 6 | 0.00 | 99.46% | **85.89%** | +13.57% | 0.04249 | 30 | 39.0s | `gs_acc85.9_h64.json` |
| #8 | #25 | 64 | 0.30 | 0.025 | 0.15 | 6 | 0.00 | 99.57% | **85.64%** | +13.93% | 0.04161 | 30 | 39.1s | `gs_acc85.6_h64.json` |
| #9 | #24 | 64 | 0.30 | 0.01 | 0.85 | 6 | 0.00 | 99.57% | **85.39%** | +14.18% | 0.04091 | 28 | 36.5s | `gs_acc85.4_h64.json` |
| #10 | #11 | 48 | 0.30 | 0.008 | 0.55 | 6 | 0.00 | 99.57% | **85.39%** | +14.18% | 0.04463 | 29 | 31.0s | `gs_acc85.4_h48.json` |

## 7. Paineis Visuais Gerados

- **Visao Global (Fronteira MSE, Overfitting e Custo):** `grid_search_insights.png`
- **Sensibilidade Parametrica Individual (Acc vs Hiperparametros):** `grid_search_hyperparams.png`
- **Mapas de Calor Bivariados (Interacao de Pares):** `grid_search_heatmaps.png` (se >= 2 variaveis)

