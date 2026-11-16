# Relatorio Analitico - Grid Search

Total de Modelos Treinados: 8


## 1. Melhor Modelo da Execucao (Top 1)

- **Rodada:** #5
- **Topologia:** `H = 48`
- **Hiperparametros:** `Val Ratio = 0.2`, `LR = 0.025`, `Momentum = 0.55`, `Patience = 10`, `Noise = 0.0`
- **Desempenho no Teste (130 fixos):** Acuracia = **82.31%** | MSE Teste = **0.04847**
- **Melhor Epoca na Validacao (Menor MSE):** Epoca 19 (MSE Val: **0.03788** | Acc Val: **86.75%**)
- **Acuracia de Treino Final:** 99.58% (Gap Treino-Val: **+12.83%**)
- **Epocas Executadas:** 29 (25.75s) | Parada Antecipada: `Sim`
- **Checkpoint Salvo:** `gs_acc82.3_h48.json`


## 2. Comparativo por Fator de Momentum

| Momentum (α) | Total Modelos | Acuracia Media | Acuracia Maxima | Gap Overfit Medio | MSE Minimo | Tempo Medio (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **α = 0.0** | 4 | 83.97% | **86.32%** | +13.31% | **0.03909** | 63.75s |
| **α = 0.55** | 4 | 84.72% | **86.75%** | +12.27% | **0.03701** | 62.79s |

## 3. Comparativo por Taxa de Ruido (Data Augmentation)

| Ruido | Total Modelos | Acuracia Media | Acuracia Maxima | Gap Overfit Medio | MSE Minimo | Tempo Medio (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0% (Limpo)** | 2 | 86.53% | **86.75%** | +13.10% | **0.03788** | 31.45s |
| **5.0%** | 2 | 85.68% | **86.32%** | +13.86% | **0.03701** | 55.32s |
| **15.0%** | 2 | 81.84% | **82.05%** | +16.24% | **0.04027** | 69.34s |
| **30.0%** | 2 | 83.34% | **84.19%** | +7.96% | **0.04106** | 96.97s |

## 4. Diagnostico de Generalizacao & Overfitting

- **Gap Medio Global (Treino - Validacao):** **+12.79%**
- **Modelo Mais Generalizador (Menor Gap):** Rodada #8 (Treino: 90.70% | Val: 84.19% | Gap: **+6.51%**)
- **Distribuicao da Capacidade de Generalizacao:**
  - Generalizacao Alta (Gap <= 5%): **0/8** modelos
  - Sobreajuste Moderado (5% < Gap <= 15%): **6/8** modelos
  - Sobreajuste Acentuado (Gap > 15%): **2/8** modelos


## 5. Top 8 Melhores Modelos da Execucao

| Pos | Rodada | H | Ratio | LR | Momentum | P | Noise | Acc Treino | Acc Val | MSE Val | Acc Teste | MSE Teste | Epocas | Tempo | Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| #1 | #5 | 48 | 0.20 | 0.025 | 0.55 | 10 | 0.00 | 99.58% | 86.75% | 0.03788 | **82.31%** | 0.04847 | 29 | 25.8s | `gs_acc82.3_h48.json` |
| #2 | #1 | 48 | 0.20 | 0.025 | 0.0 | 10 | 0.00 | 99.69% | 86.32% | 0.03909 | **81.54%** | 0.05204 | 42 | 37.1s | `gs_acc81.5_h48.json` |
| #3 | #7 | 48 | 0.20 | 0.025 | 0.55 | 10 | 0.15 | 98.28% | 81.62% | 0.04027 | **81.54%** | 0.05145 | 53 | 92.0s | `gs_acc81.5_h48_v2.json` |
| #4 | #6 | 48 | 0.20 | 0.025 | 0.55 | 10 | 0.05 | 99.38% | 86.32% | 0.03701 | **80.77%** | 0.04962 | 37 | 64.0s | `gs_acc80.8_h48.json` |
| #5 | #2 | 48 | 0.20 | 0.025 | 0.0 | 10 | 0.05 | 99.69% | 85.04% | 0.03990 | **80.00%** | 0.05148 | 27 | 46.7s | `gs_acc80.0_h48.json` |
| #6 | #3 | 48 | 0.20 | 0.025 | 0.0 | 10 | 0.15 | 97.87% | 82.05% | 0.04457 | **79.23%** | 0.05325 | 27 | 46.6s | `gs_acc79.2_h48.json` |
| #7 | #8 | 48 | 0.20 | 0.025 | 0.55 | 10 | 0.30 | 90.70% | 84.19% | 0.04261 | **75.38%** | 0.06683 | 40 | 69.4s | `gs_acc75.4_h48.json` |
| #8 | #4 | 48 | 0.20 | 0.025 | 0.0 | 10 | 0.30 | 91.89% | 82.48% | 0.04106 | **72.31%** | 0.06744 | 72 | 124.5s | `gs_acc72.3_h48.json` |

## 6. Paineis Visuais Gerados

- **Visao Global (Fronteira MSE, Overfitting e Custo):** `grid_search_insights.png`
- **Sensibilidade Parametrica Individual (Acc vs Hiperparametros):** `grid_search_hyperparams.png`
- **Mapas de Calor Bivariados (Interacao de Pares):** `grid_search_heatmaps.png` (se >= 2 variaveis)

