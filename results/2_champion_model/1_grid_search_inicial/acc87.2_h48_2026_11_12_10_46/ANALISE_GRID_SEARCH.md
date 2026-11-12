# Relatorio Analitico - Grid Search

Total de Modelos Treinados: 27  

---

## 1. Melhor Modelo da Execucao

- **Rodada:** #17
- **Topologia:** `H = 48`
- **Hiperparametros:** `LR = 0.025`, `Momentum = 0.55`, `Patience = 6`
- **Melhor Acuracia de Validacao:** **87.15%**
- **Menor MSE de Validacao:** **0.03918**
- **Epocas Executadas:** 20 (convergencia em 20.28s)
- **Checkpoint Salvo:** `gs_acc87.2_h48.json`

---

## 2. Comparativo por Topologia (H)

| Neuronios (H) | Total Modelos | Acuracia Media | Acuracia Maxima | MSE Minimo | Tempo Medio (s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **H=16** | 9 | 71.12% | **75.57%** | **0.06109** | 14.14s |
| **H=48** | 9 | 85.05% | **87.15%** | **0.03865** | 26.41s |
| **H=64** | 9 | 85.08% | **86.40%** | **0.04029** | 42.83s |

---

## 3. Top 5 Melhores Modelos

| Posicao | Rodada | H | LR | Momentum | Acc Val | MSE Val | Epocas | Tempo | Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| #1 | Rodada 17 | H=48 | 0.025 | 0.55 | **87.15%** | 0.03918 | 20 | 20.3s | `gs_acc87.2_h48.json` |
| #2 | Rodada 12 | H=48 | 0.008 | 0.85 | **86.40%** | 0.03865 | 29 | 30.4s | `gs_acc86.4_h48.json` |
| #3 | Rodada 15 | H=48 | 0.01 | 0.85 | **86.40%** | 0.03876 | 26 | 26.9s | `gs_acc86.4_h48_v2.json` |
| #4 | Rodada 26 | H=64 | 0.025 | 0.55 | **86.40%** | 0.04029 | 35 | 45.5s | `gs_acc86.4_h64.json` |
| #5 | Rodada 21 | H=64 | 0.008 | 0.85 | **86.15%** | 0.04067 | 30 | 39.3s | `gs_acc86.1_h64.json` |
