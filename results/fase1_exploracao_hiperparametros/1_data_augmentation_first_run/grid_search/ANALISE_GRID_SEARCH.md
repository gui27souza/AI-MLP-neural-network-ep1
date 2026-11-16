# Relatorio Analitico - Grid Search

Total de Modelos Treinados: 27  

---

## 1. Melhor Modelo da Execucao

- **Rodada:** #18
- **Topologia:** `H = 48`
- **Hiperparametros:** `LR = 0.025`, `Momentum = 0.85`, `Patience = 6`
- **Melhor Acuracia de Validacao:** **85.89%**
- **Menor MSE de Validacao:** **0.04437**
- **Epocas Executadas:** 31 (convergencia em 64.19s)
- **Checkpoint Salvo:** `gs_acc85.9_h48.json`

---

## 2. Comparativo por Topologia (H)

| Neuronios (H) | Total Modelos | Acuracia Media | Acuracia Maxima | MSE Minimo | Tempo Medio (s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **H=16** | 9 | 71.65% | **74.31%** | **0.06196** | 16.44s |
| **H=48** | 9 | 84.30% | **85.89%** | **0.04144** | 46.31s |
| **H=64** | 9 | 83.88% | **85.14%** | **0.04049** | 58.19s |

---

## 3. Top 5 Melhores Modelos

| Posicao | Rodada | H | LR | Momentum | Acc Val | MSE Val | Epocas | Tempo | Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| #1 | Rodada 18 | H=48 | 0.025 | 0.85 | **85.89%** | 0.04437 | 31 | 64.2s | `gs_acc85.9_h48.json` |
| #2 | Rodada 24 | H=64 | 0.01 | 0.85 | **85.14%** | 0.04049 | 17 | 48.0s | `gs_acc85.1_h64.json` |
| #3 | Rodada 21 | H=64 | 0.008 | 0.85 | **84.63%** | 0.04123 | 24 | 64.7s | `gs_acc84.6_h64.json` |
| #4 | Rodada 14 | H=48 | 0.01 | 0.55 | **84.63%** | 0.04317 | 26 | 53.3s | `gs_acc84.6_h48_v2.json` |
| #5 | Rodada 13 | H=48 | 0.01 | 0.15 | **84.63%** | 0.04531 | 24 | 49.6s | `gs_acc84.6_h48.json` |
