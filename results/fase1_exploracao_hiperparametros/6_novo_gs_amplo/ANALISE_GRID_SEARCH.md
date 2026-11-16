# Relatorio Analitico - Grid Search

Total de Modelos Treinados: 384


## 1. Melhor Modelo da Execucao (Top 1)

- **Rodada:** #265
- **Topologia:** `H = 48`
- **Hiperparametros:** `Val Ratio = 0.3`, `LR = 0.07`, `Momentum = 0.0`, `Patience = 14`, `Noise = 0.0`
- **Melhor Acuracia de Validacao:** **87.15%** (alcancada na epoca 19)
- **Acuracia de Treino Final:** 99.68% (Gap Treino-Val: **+12.53%**)
- **Menor MSE de Validacao:** **0.03946**
- **Epocas Executadas:** 33 (31.45s) | Parada Antecipada: `Sim`
- **Checkpoint Salvo:** `gs_acc87.2_h48.json`


## 2. Comparativo por Topologia (Neuronios Ocultos H)

| Neuronios (H) | Total Modelos | Acuracia Media | Acuracia Maxima | Gap Overfit Medio | MSE Minimo | Tempo Medio (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H=24** | 96 | 68.09% | **81.11%** | +17.80% | **0.04909** | 37.32s |
| **H=48** | 96 | 71.86% | **87.15%** | +14.57% | **0.03902** | 69.40s |
| **H=96** | 96 | 67.20% | **85.14%** | +13.69% | **0.03980** | 164.04s |
| **H=144** | 96 | 63.68% | **85.89%** | +11.80% | **0.04038** | 243.60s |

## 3. Comparativo por Proporcao de Validacao (Val Ratio)

| Val Ratio | Total Modelos | Acuracia Media | Acuracia Maxima | Gap Overfit Medio | MSE Minimo | Tempo Medio (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **15.0% (0.15)** | 192 | 66.19% | **84.34%** | +15.03% | **0.04540** | 119.08s |
| **30.0% (0.30)** | 192 | 69.22% | **87.15%** | +13.90% | **0.03902** | 138.10s |

## 4. Comparativo por Taxa de Aprendizado (Learn Rate)

| Taxa (η) | Total Modelos | Acuracia Media | Acuracia Maxima | Gap Overfit Medio | MSE Minimo | Tempo Medio (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **η = 0.002** | 96 | 81.04% | **85.64%** | +17.18% | **0.04192** | 165.04s |
| **η = 0.02** | 96 | 77.68% | **86.65%** | +17.09% | **0.03902** | 117.19s |
| **η = 0.07** | 96 | 59.77% | **87.15%** | +12.65% | **0.03946** | 120.97s |
| **η = 0.1** | 96 | 52.33% | **84.89%** | +10.94% | **0.03980** | 111.16s |

## 5. Comparativo por Fator de Momentum

| Momentum (α) | Total Modelos | Acuracia Media | Acuracia Maxima | Gap Overfit Medio | MSE Minimo | Tempo Medio (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **α = 0.0** | 96 | 80.95% | **87.15%** | +17.56% | **0.03946** | 124.27s |
| **α = 0.4** | 96 | 80.46% | **86.15%** | +17.87% | **0.04074** | 132.00s |
| **α = 0.7** | 96 | 67.82% | **86.65%** | +14.90% | **0.03902** | 134.83s |
| **α = 0.9** | 96 | 41.60% | **85.64%** | +7.53% | **0.04192** | 123.27s |

## 6. Comparativo por Taxa de Ruido (Data Augmentation)

| Ruido | Total Modelos | Acuracia Media | Acuracia Maxima | Gap Overfit Medio | MSE Minimo | Tempo Medio (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0% (Limpo)** | 128 | 68.03% | **87.15%** | +13.94% | **0.03902** | 88.19s |
| **3.0%** | 128 | 68.37% | **85.89%** | +14.61% | **0.03999** | 153.65s |
| **8.0%** | 128 | 66.71% | **84.63%** | +14.85% | **0.04188** | 143.93s |

## 7. Diagnostico de Generalizacao & Overfitting

- **Gap Medio Global (Treino - Validacao):** **+14.46%**
- **Modelo Mais Generalizador (Menor Gap):** Rodada #384 (Treino: 7.05% | Val: 7.05% | Gap: **+0.00%**)
- **Distribuicao da Capacidade de Generalizacao:**
  - Generalizacao Alta (Gap <= 5%): **64/384** modelos
  - Sobreajuste Moderado (5% < Gap <= 15%): **59/384** modelos
  - Sobreajuste Acentuado (Gap > 15%): **261/384** modelos


## 8. Top 10 Melhores Modelos da Execucao

| Pos | Rodada | H | Ratio | LR | Momentum | P | Noise | Acc Treino | Acc Val | Gap Overfit | MSE Val | Epocas | Tempo | Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| #1 | #265 | 48 | 0.30 | 0.07 | 0.0 | 14 | 0.00 | 99.68% | **87.15%** | +12.53% | 0.03946 | 33 | 31.4s | `gs_acc87.2_h48.json` |
| #2 | #259 | 48 | 0.30 | 0.02 | 0.7 | 14 | 0.00 | 99.68% | **86.65%** | +13.03% | 0.03902 | 38 | 36.1s | `gs_acc86.6_h48.json` |
| #3 | #256 | 48 | 0.30 | 0.02 | 0.4 | 14 | 0.00 | 99.68% | **86.15%** | +13.53% | 0.04077 | 38 | 36.1s | `gs_acc86.1_h48.json` |
| #4 | #356 | 144 | 0.30 | 0.02 | 0.7 | 14 | 0.03 | 99.30% | **85.89%** | +13.41% | 0.04038 | 44 | 243.5s | `gs_acc85.9_h144.json` |
| #5 | #253 | 48 | 0.30 | 0.02 | 0.0 | 14 | 0.00 | 99.57% | **85.89%** | +13.68% | 0.04293 | 37 | 35.2s | `gs_acc85.9_h48.json` |
| #6 | #362 | 144 | 0.30 | 0.07 | 0.0 | 14 | 0.03 | 99.30% | **85.64%** | +13.66% | 0.04064 | 47 | 260.2s | `gs_acc85.6_h144_v2.json` |
| #7 | #346 | 144 | 0.30 | 0.002 | 0.9 | 14 | 0.00 | 99.46% | **85.64%** | +13.82% | 0.04346 | 49 | 140.5s | `gs_acc85.6_h144.json` |
| #8 | #352 | 144 | 0.30 | 0.02 | 0.4 | 14 | 0.00 | 99.35% | **85.39%** | +13.96% | 0.04229 | 35 | 100.2s | `gs_acc85.4_h144_v2.json` |
| #9 | #349 | 144 | 0.30 | 0.02 | 0.0 | 14 | 0.00 | 99.35% | **85.39%** | +13.96% | 0.04346 | 49 | 140.3s | `gs_acc85.4_h144.json` |
| #10 | #266 | 48 | 0.30 | 0.07 | 0.0 | 14 | 0.03 | 99.57% | **85.14%** | +14.43% | 0.04066 | 30 | 54.8s | `gs_acc85.1_h48_v4.json` |

## 9. Paineis Visuais Gerados

- **Visao Global (Fronteira MSE, Overfitting e Custo):** `grid_search_insights.png`
- **Sensibilidade Parametrica Individual (Acc vs Hiperparametros):** `grid_search_hyperparams.png`
- **Mapas de Calor Bivariados (Interacao de Pares):** `grid_search_heatmaps.png` (se >= 2 variaveis)

