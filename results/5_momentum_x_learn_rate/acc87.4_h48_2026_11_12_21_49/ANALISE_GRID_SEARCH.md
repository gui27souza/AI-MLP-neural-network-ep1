# Relatorio Analitico - Grid Search

Total de Modelos Treinados: 9


## 1. Melhor Modelo da Execucao (Top 1)

- **Rodada:** #5
- **Topologia:** `H = 48`
- **Hiperparametros:** `Val Ratio = 0.3`, `LR = 0.025`, `Momentum = 0.55`, `Patience = 10`, `Noise = 0.0`
- **Melhor Acuracia de Validacao:** **87.41%** (alcancada na epoca 22)
- **Acuracia de Treino Final:** 99.68% (Gap Treino-Val: **+12.27%**)
- **Menor MSE de Validacao:** **0.03852**
- **Epocas Executadas:** 32 (30.96s) | Parada Antecipada: `Sim`
- **Checkpoint Salvo:** `gs_acc87.4_h48.json`


## 2. Comparativo por Taxa de Aprendizado (Learn Rate)

| Taxa (η) | Total Modelos | Acuracia Media | Acuracia Maxima | Gap Overfit Medio | MSE Minimo | Tempo Medio (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **η = 0.0** | 3 | 2.52% | **2.52%** | +0.39% | **0.93920** | 10.64s |
| **η = 0.025** | 3 | 82.29% | **87.41%** | +14.81% | **0.03852** | 40.25s |
| **η = 0.1** | 3 | 56.51% | **84.89%** | +11.67% | **0.04106** | 28.18s |

## 3. Comparativo por Fator de Momentum

| Momentum (α) | Total Modelos | Acuracia Media | Acuracia Maxima | Gap Overfit Medio | MSE Minimo | Tempo Medio (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **α = 0.0** | 3 | 57.60% | **85.39%** | +9.68% | **0.04106** | 21.93s |
| **α = 0.55** | 3 | 55.84% | **87.41%** | +11.30% | **0.03852** | 30.42s |
| **α = 0.9** | 3 | 27.88% | **74.06%** | +5.89% | **0.07223** | 26.72s |

## 4. Diagnostico de Generalizacao & Overfitting

- **Gap Medio Global (Treino - Validacao):** **+8.95%**
- **Modelo Mais Generalizador (Menor Gap):** Rodada #1 (Treino: 2.91% | Val: 2.52% | Gap: **+0.39%**)
- **Distribuicao da Capacidade de Generalizacao:**
  - Generalizacao Alta (Gap <= 5%): **4/9** modelos
  - Sobreajuste Moderado (5% < Gap <= 15%): **3/9** modelos
  - Sobreajuste Acentuado (Gap > 15%): **2/9** modelos


## 5. Top 9 Melhores Modelos da Execucao

| Pos | Rodada | H | Ratio | LR | Momentum | P | Noise | Acc Treino | Acc Val | Gap Overfit | MSE Val | Epocas | Tempo | Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| #1 | #5 | 48 | 0.30 | 0.025 | 0.55 | 10 | 0.00 | 99.68% | **87.41%** | +12.27% | 0.03852 | 32 | 31.0s | `gs_acc87.4_h48.json` |
| #2 | #4 | 48 | 0.30 | 0.025 | 0.0 | 10 | 0.00 | 99.57% | **85.39%** | +14.18% | 0.04243 | 33 | 31.8s | `gs_acc85.4_h48.json` |
| #3 | #7 | 48 | 0.30 | 0.1 | 0.0 | 10 | 0.00 | 99.35% | **84.89%** | +14.46% | 0.04106 | 24 | 23.3s | `gs_acc84.9_h48.json` |
| #4 | #8 | 48 | 0.30 | 0.1 | 0.55 | 10 | 0.00 | 98.82% | **77.58%** | +21.24% | 0.06457 | 51 | 49.7s | `gs_acc77.6_h48.json` |
| #5 | #6 | 48 | 0.30 | 0.025 | 0.9 | 10 | 0.00 | 92.03% | **74.06%** | +17.97% | 0.07223 | 60 | 58.0s | `` |
| #6 | #9 | 48 | 0.30 | 0.1 | 0.9 | 10 | 0.00 | 6.35% | **7.05%** | -0.70% | 0.15100 | 12 | 11.6s | `` |
| #7 | #1 | 48 | 0.30 | 0.0 | 0.0 | 10 | 0.00 | 2.91% | **2.52%** | +0.39% | 0.93920 | 11 | 10.7s | `` |
| #8 | #2 | 48 | 0.30 | 0.0 | 0.55 | 10 | 0.00 | 2.91% | **2.52%** | +0.39% | 0.93920 | 11 | 10.6s | `` |
| #9 | #3 | 48 | 0.30 | 0.0 | 0.9 | 10 | 0.00 | 2.91% | **2.52%** | +0.39% | 0.93920 | 11 | 10.6s | `` |

## 6. Paineis Visuais Gerados

- **Visao Global (Fronteira MSE, Overfitting e Custo):** `grid_search_insights.png`
- **Sensibilidade Parametrica Individual (Acc vs Hiperparametros):** `grid_search_hyperparams.png`
- **Mapas de Calor Bivariados (Interacao de Pares):** `grid_search_heatmaps.png` (se >= 2 variaveis)

