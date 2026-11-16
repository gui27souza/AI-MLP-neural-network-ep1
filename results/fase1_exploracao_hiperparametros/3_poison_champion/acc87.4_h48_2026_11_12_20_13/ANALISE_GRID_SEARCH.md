# Relatorio Analitico - Grid Search

Total de Modelos Treinados: 5

## 1. Melhor Modelo da Execucao

- **Rodada:** #1
- **Topologia:** `H = 48`
- **Hiperparametros:** `LR = 0.025`, `Momentum = 0.55`, `Patience = 10`, `Noise = 0.0`
- **Melhor Acuracia de Validacao:** **87.41%**
- **Menor MSE de Validacao:** **0.03852**
- **Epocas Executadas:** 32 (31.01s)
- **Checkpoint Salvo:** `gs_acc87.4_h48.json`

## 2. Comparativo por Topologia (H)

| Neuronios (H) | Total Modelos | Acuracia Media | Acuracia Maxima | MSE Minimo | Tempo Medio (s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **H=48** | 5 | 79.65% | **87.41%** | **0.03852** | 50.80s |

## 3. Comparativo por Taxa de Ruido (Data Augmentation)

| Taxa de Ruido | Total Modelos | Acuracia Media | Acuracia Maxima | MSE Minimo | Tempo Medio (s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0% (Limpo)** | 1 | 87.41% | **87.41%** | **0.03852** | 31.01s |
| **3.0%** | 1 | 84.38% | **84.38%** | **0.04280** | 41.16s |
| **15.0%** | 1 | 82.12% | **82.12%** | **0.04595** | 56.08s |
| **50.0%** | 1 | 82.37% | **82.37%** | **0.05230** | 71.27s |
| **90.0%** | 1 | 61.96% | **61.96%** | **0.07301** | 54.48s |

## 4. Top 5 Melhores Modelos

| Posicao | Rodada | H | LR | Momentum | Noise | Acc Val | MSE Val | Epocas | Tempo | Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| #1 | Rodada 1 | H=48 | 0.025 | 0.55 | 0.00 | **87.41%** | 0.03852 | 32 | 31.0s | `gs_acc87.4_h48.json` |
| #2 | Rodada 2 | H=48 | 0.025 | 0.55 | 0.03 | **84.38%** | 0.04280 | 22 | 41.2s | `gs_acc84.4_h48.json` |
| #3 | Rodada 4 | H=48 | 0.025 | 0.55 | 0.50 | **82.37%** | 0.05230 | 38 | 71.3s | `gs_acc82.4_h48.json` |
| #4 | Rodada 3 | H=48 | 0.025 | 0.55 | 0.15 | **82.12%** | 0.04595 | 30 | 56.1s | `gs_acc82.1_h48.json` |
| #5 | Rodada 5 | H=48 | 0.025 | 0.55 | 0.90 | **61.96%** | 0.07301 | 29 | 54.5s | `` |
