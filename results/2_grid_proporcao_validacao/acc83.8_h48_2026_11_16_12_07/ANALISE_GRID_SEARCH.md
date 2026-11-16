# Relatorio Analitico - Grid Search

Total de Modelos Treinados: 12


## 1. Melhor Modelo da Execucao (Top 1)

- **Rodada:** #5
- **Topologia:** `H = 48`
- **Hiperparametros:** `Val Ratio = 0.2`, `LR = 0.015`, `Momentum = 0.0`, `Patience = 10`, `Noise = 0.0`
- **Desempenho no Teste (130 fixos):** Acuracia = **83.85%** | MSE Teste = **0.05234**
- **Melhor Epoca na Validacao (Menor MSE):** Epoca 55 (MSE Val: **0.04042** | Acc Val: **86.75%**)
- **Acuracia de Treino Final:** 99.79% (Gap Treino-Val: **+13.04%**)
- **Epocas Executadas:** 65 (57.37s) | Parada Antecipada: `Sim`
- **Checkpoint Salvo:** `gs_acc83.8_h48.json`


## 2. Comparativo por Proporcao de Validacao (Val Ratio)

| Val Ratio | Total Modelos | Acuracia Media | Acuracia Maxima | Gap Overfit Medio | MSE Minimo | Tempo Medio (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10.0% (0.10)** | 4 | 81.54% | **83.08%** | +18.01% | **0.04443** | 33.45s |
| **20.0% (0.20)** | 4 | 86.64% | **86.75%** | +13.07% | **0.03788** | 39.37s |
| **30.0% (0.30)** | 4 | 87.22% | **87.91%** | +12.50% | **0.03335** | 36.79s |

## 3. Comparativo por Taxa de Aprendizado (Learn Rate)

| Taxa (η) | Total Modelos | Acuracia Media | Acuracia Maxima | Gap Overfit Medio | MSE Minimo | Tempo Medio (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **η = 0.015** | 6 | 85.36% | **87.91%** | +14.29% | **0.03335** | 40.63s |
| **η = 0.025** | 6 | 84.92% | **87.36%** | +14.77% | **0.03428** | 32.44s |

## 4. Comparativo por Fator de Momentum

| Momentum (α) | Total Modelos | Acuracia Media | Acuracia Maxima | Gap Overfit Medio | MSE Minimo | Tempo Medio (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **α = 0.0** | 6 | 85.32% | **87.64%** | +14.35% | **0.03434** | 42.99s |
| **α = 0.55** | 6 | 84.95% | **87.91%** | +14.71% | **0.03335** | 30.08s |

## 5. Diagnostico de Generalizacao & Overfitting

- **Gap Medio Global (Treino - Validacao):** **+14.53%**
- **Modelo Mais Generalizador (Menor Gap):** Rodada #10 (Treino: 99.76% | Val: 87.91% | Gap: **+11.85%**)
- **Distribuicao da Capacidade de Generalizacao:**
  - Generalizacao Alta (Gap <= 5%): **0/12** modelos
  - Sobreajuste Moderado (5% < Gap <= 15%): **8/12** modelos
  - Sobreajuste Acentuado (Gap > 15%): **4/12** modelos


## 6. Top 10 Melhores Modelos da Execucao

| Pos | Rodada | H | Ratio | LR | Momentum | P | Noise | Acc Treino | Acc Val | MSE Val | Acc Teste | MSE Teste | Epocas | Tempo | Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| #1 | #5 | 48 | 0.20 | 0.015 | 0.0 | 10 | 0.00 | 99.79% | 86.75% | 0.04042 | **83.85%** | 0.05234 | 65 | 57.4s | `gs_acc83.8_h48.json` |
| #2 | #2 | 48 | 0.10 | 0.015 | 0.55 | 10 | 0.00 | 99.34% | 80.00% | 0.04666 | **83.08%** | 0.04894 | 20 | 19.1s | `gs_acc83.1_h48.json` |
| #3 | #10 | 48 | 0.30 | 0.015 | 0.55 | 10 | 0.00 | 99.76% | 87.91% | 0.03335 | **82.31%** | 0.05416 | 50 | 39.9s | `gs_acc82.3_h48_v4.json` |
| #4 | #12 | 48 | 0.30 | 0.025 | 0.55 | 10 | 0.00 | 99.76% | 85.99% | 0.03428 | **82.31%** | 0.05277 | 34 | 27.1s | `gs_acc82.3_h48_v5.json` |
| #5 | #9 | 48 | 0.30 | 0.015 | 0.0 | 10 | 0.00 | 99.64% | 87.64% | 0.03459 | **82.31%** | 0.05588 | 50 | 39.7s | `gs_acc82.3_h48_v3.json` |
| #6 | #8 | 48 | 0.20 | 0.025 | 0.55 | 10 | 0.00 | 99.58% | 86.75% | 0.03788 | **82.31%** | 0.04847 | 29 | 25.7s | `gs_acc82.3_h48_v2.json` |
| #7 | #6 | 48 | 0.20 | 0.015 | 0.55 | 10 | 0.00 | 99.79% | 86.75% | 0.03839 | **82.31%** | 0.05088 | 42 | 37.2s | `gs_acc82.3_h48.json` |
| #8 | #7 | 48 | 0.20 | 0.025 | 0.0 | 10 | 0.00 | 99.69% | 86.32% | 0.03909 | **81.54%** | 0.05204 | 42 | 37.2s | `gs_acc81.5_h48_v3.json` |
| #9 | #1 | 48 | 0.10 | 0.015 | 0.0 | 10 | 0.00 | 99.53% | 83.08% | 0.04560 | **81.54%** | 0.05298 | 53 | 50.6s | `gs_acc81.5_h48.json` |
| #10 | #3 | 48 | 0.10 | 0.025 | 0.0 | 10 | 0.00 | 99.62% | 80.77% | 0.04766 | **81.54%** | 0.05138 | 34 | 32.6s | `gs_acc81.5_h48_v2.json` |

## 7. Paineis Visuais Gerados

- **Visao Global (Fronteira MSE, Overfitting e Custo):** `grid_search_insights.png`
- **Sensibilidade Parametrica Individual (Acc vs Hiperparametros):** `grid_search_hyperparams.png`
- **Mapas de Calor Bivariados (Interacao de Pares):** `grid_search_heatmaps.png` (se >= 2 variaveis)

