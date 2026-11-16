# Relatorio Analitico - Grid Search

Total de Modelos Treinados: 27


## 1. Melhor Modelo da Execucao (Top 1)

- **Rodada:** #20
- **Topologia:** `H = 64`
- **Hiperparametros:** `Val Ratio = 0.2`, `LR = 0.008`, `Momentum = 0.55`, `Patience = 10`, `Noise = 0.0`
- **Desempenho no Teste (130 fixos):** Acuracia = **84.62%** | MSE Teste = **0.05621**
- **Melhor Epoca na Validacao (Menor MSE):** Epoca 20 (MSE Val: **0.04004** | Acc Val: **84.62%**)
- **Acuracia de Treino Final:** 99.38% (Gap Treino-Val: **+14.76%**)
- **Epocas Executadas:** 30 (35.04s) | Parada Antecipada: `Sim`
- **Checkpoint Salvo:** `gs_acc84.6_h64.json`


## 2. Comparativo por Topologia (Neuronios Ocultos H)

| Neuronios (H) | Total Modelos | Acuracia Media | Acuracia Maxima | Gap Overfit Medio | MSE Minimo | Tempo Medio (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H=16** | 9 | 76.45% | **79.06%** | +20.51% | **0.05571** | 15.69s |
| **H=48** | 9 | 85.80% | **86.75%** | +13.80% | **0.03768** | 38.20s |
| **H=64** | 9 | 84.76% | **85.04%** | +14.70% | **0.03582** | 44.80s |

## 3. Comparativo por Taxa de Aprendizado (Learn Rate)

| Taxa (η) | Total Modelos | Acuracia Media | Acuracia Maxima | Gap Overfit Medio | MSE Minimo | Tempo Medio (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **η = 0.008** | 9 | 82.62% | **86.32%** | +16.26% | **0.03723** | 39.38s |
| **η = 0.015** | 9 | 82.57% | **86.75%** | +15.96% | **0.03582** | 31.47s |
| **η = 0.025** | 9 | 81.81% | **86.75%** | +16.79% | **0.03724** | 27.83s |

## 4. Comparativo por Fator de Momentum

| Momentum (α) | Total Modelos | Acuracia Media | Acuracia Maxima | Gap Overfit Medio | MSE Minimo | Tempo Medio (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **α = 0.0** | 9 | 82.57% | **86.75%** | +16.40% | **0.03856** | 47.46s |
| **α = 0.55** | 9 | 83.00% | **86.75%** | +15.99% | **0.03724** | 27.22s |
| **α = 0.85** | 9 | 81.43% | **86.32%** | +16.62% | **0.03582** | 24.00s |

## 5. Diagnostico de Generalizacao & Overfitting

- **Gap Medio Global (Treino - Validacao):** **+16.34%**
- **Modelo Mais Generalizador (Menor Gap):** Rodada #17 (Treino: 99.58% | Val: 86.75% | Gap: **+12.83%**)
- **Distribuicao da Capacidade de Generalizacao:**
  - Generalizacao Alta (Gap <= 5%): **0/27** modelos
  - Sobreajuste Moderado (5% < Gap <= 15%): **15/27** modelos
  - Sobreajuste Acentuado (Gap > 15%): **12/27** modelos


## 6. Top 10 Melhores Modelos da Execucao

| Pos | Rodada | H | Ratio | LR | Momentum | P | Noise | Acc Treino | Acc Val | MSE Val | Acc Teste | MSE Teste | Epocas | Tempo | Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| #1 | #20 | 64 | 0.20 | 0.008 | 0.55 | 10 | 0.00 | 99.38% | 84.62% | 0.04004 | **84.62%** | 0.05621 | 30 | 35.0s | `gs_acc84.6_h64.json` |
| #2 | #12 | 48 | 0.20 | 0.008 | 0.85 | 10 | 0.00 | 99.79% | 86.32% | 0.03787 | **83.85%** | 0.04972 | 31 | 27.4s | `gs_acc83.8_h48.json` |
| #3 | #13 | 48 | 0.20 | 0.015 | 0.0 | 10 | 0.00 | 99.79% | 86.75% | 0.04042 | **83.85%** | 0.05234 | 65 | 57.4s | `gs_acc83.8_h48_v2.json` |
| #4 | #11 | 48 | 0.20 | 0.008 | 0.55 | 10 | 0.00 | 99.69% | 86.32% | 0.04101 | **83.08%** | 0.05162 | 42 | 37.1s | `gs_acc83.1_h48_v2.json` |
| #5 | #10 | 48 | 0.20 | 0.008 | 0.0 | 10 | 0.00 | 99.48% | 85.90% | 0.04155 | **83.08%** | 0.05228 | 79 | 69.7s | `gs_acc83.1_h48.json` |
| #6 | #17 | 48 | 0.20 | 0.025 | 0.55 | 10 | 0.00 | 99.58% | 86.75% | 0.03788 | **82.31%** | 0.04847 | 29 | 25.6s | `gs_acc82.3_h48_v2.json` |
| #7 | #14 | 48 | 0.20 | 0.015 | 0.55 | 10 | 0.00 | 99.79% | 86.75% | 0.03839 | **82.31%** | 0.05088 | 42 | 37.2s | `gs_acc82.3_h48.json` |
| #8 | #26 | 64 | 0.20 | 0.025 | 0.55 | 10 | 0.00 | 99.69% | 85.04% | 0.03724 | **81.54%** | 0.05163 | 33 | 38.6s | `gs_acc81.5_h64.json` |
| #9 | #16 | 48 | 0.20 | 0.025 | 0.0 | 10 | 0.00 | 99.69% | 86.32% | 0.03909 | **81.54%** | 0.05204 | 42 | 37.2s | `gs_acc81.5_h48.json` |
| #10 | #24 | 64 | 0.20 | 0.015 | 0.85 | 10 | 0.00 | 99.58% | 85.04% | 0.03582 | **80.77%** | 0.05147 | 23 | 26.9s | `gs_acc80.8_h64.json` |

## 7. Paineis Visuais Gerados

- **Visao Global (Fronteira MSE, Overfitting e Custo):** `grid_search_insights.png`
- **Sensibilidade Parametrica Individual (Acc vs Hiperparametros):** `grid_search_hyperparams.png`
- **Mapas de Calor Bivariados (Interacao de Pares):** `grid_search_heatmaps.png` (se >= 2 variaveis)

