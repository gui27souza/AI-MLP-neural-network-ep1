# Relatorio Analitico - Grid Search

Total de Modelos Treinados: 5


## 1. Melhor Modelo da Execucao (Top 1)

- **Rodada:** #3
- **Topologia:** `H = 48`
- **Hiperparametros:** `Val Ratio = 0.3`, `LR = 0.025`, `Momentum = 0.55`, `Patience = 10`, `Noise = 0.0`
- **Melhor Acuracia de Validacao:** **87.41%** (alcancada na epoca 22)
- **Acuracia de Treino Final:** 99.68% (Gap Treino-Val: **+12.27%**)
- **Menor MSE de Validacao:** **0.03852**
- **Epocas Executadas:** 32 (30.87s) | Parada Antecipada: `Sim`
- **Checkpoint Salvo:** `gs_acc87.4_h48.json`


## 2. Comparativo por Proporcao de Validacao (Val Ratio)

| Val Ratio | Total Modelos | Acuracia Media | Acuracia Maxima | Gap Overfit Medio | MSE Minimo | Tempo Medio (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5.0% (0.05)** | 1 | 83.33% | **83.33%** | +15.64% | **0.04472** | 30.69s |
| **10.0% (0.10)** | 1 | 83.33% | **83.33%** | +15.75% | **0.04681** | 36.47s |
| **30.0% (0.30)** | 1 | 87.41% | **87.41%** | +12.27% | **0.03852** | 30.87s |
| **45.0% (0.45)** | 1 | 79.53% | **79.53%** | +19.65% | **0.05176** | 19.33s |
| **60.0% (0.60)** | 1 | 77.74% | **77.74%** | +21.51% | **0.05258** | 20.80s |

## 3. Diagnostico de Generalizacao & Overfitting

- **Gap Medio Global (Treino - Validacao):** **+16.96%**
- **Modelo Mais Generalizador (Menor Gap):** Rodada #3 (Treino: 99.68% | Val: 87.41% | Gap: **+12.27%**)
- **Distribuicao da Capacidade de Generalizacao:**
  - Generalizacao Alta (Gap <= 5%): **0/5** modelos
  - Sobreajuste Moderado (5% < Gap <= 15%): **1/5** modelos
  - Sobreajuste Acentuado (Gap > 15%): **4/5** modelos


## 4. Top 5 Melhores Modelos da Execucao

| Pos | Rodada | H | Ratio | LR | Momentum | P | Noise | Acc Treino | Acc Val | Gap Overfit | MSE Val | Epocas | Tempo | Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| #1 | #3 | 48 | 0.30 | 0.025 | 0.55 | 10 | 0.00 | 99.68% | **87.41%** | +12.27% | 0.03852 | 32 | 30.9s | `gs_acc87.4_h48.json` |
| #2 | #1 | 48 | 0.05 | 0.025 | 0.55 | 10 | 0.00 | 98.97% | **83.33%** | +15.64% | 0.04472 | 25 | 30.7s | `gs_acc83.3_h48.json` |
| #3 | #2 | 48 | 0.10 | 0.025 | 0.55 | 10 | 0.00 | 99.08% | **83.33%** | +15.75% | 0.04681 | 31 | 36.5s | `gs_acc83.3_h48_v2.json` |
| #4 | #4 | 48 | 0.45 | 0.025 | 0.55 | 10 | 0.00 | 99.18% | **79.53%** | +19.65% | 0.05176 | 24 | 19.3s | `gs_acc79.5_h48.json` |
| #5 | #5 | 48 | 0.60 | 0.025 | 0.55 | 10 | 0.00 | 99.25% | **77.74%** | +21.51% | 0.05258 | 32 | 20.8s | `gs_acc77.7_h48.json` |

## 5. Paineis Visuais Gerados

- **Visao Global (Fronteira MSE, Overfitting e Custo):** `grid_search_insights.png`
- **Sensibilidade Parametrica Individual (Acc vs Hiperparametros):** `grid_search_hyperparams.png`
- **Mapas de Calor Bivariados (Interacao de Pares):** `grid_search_heatmaps.png` (se >= 2 variaveis)

