# Relatorio Analitico e Visual — Grid Search

- **Total de Modelos Treinados:** `27` combinacoes
- **Esforco Computacional Total:** `1186` epocas acumuladas em `888.1s` (`14.8 min`)
- **Conjunto de Avaliacao Final:** `130 amostras fixas de Teste` (ultimos 5 alfabetos completos, indices `1196..1325`)


## 1. Destaques da Execucao (Campeoes por Criterio)

| Criterio | Rodada | Configuracao | Acc Teste (130 fixos) | MSE Teste | Acc Val (Melhor Ep) | MSE Val | Epocas / Tempo | Checkpoint |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Campeao no Teste (Maior Acc)** | #20 | `H=64, LR=0.008, M=0.55, Val=0.2, P=10, Noise=0.0` | **84.62%** | **0.05621** | 84.62% (Ep 20) | 0.04004 | 30 ep (35.0s) | `gs_acc84.6_h64.json` |
| **Campeao na Validacao (Menor MSE)** | #24 | `H=64, LR=0.015, M=0.85, Val=0.2, P=10, Noise=0.0` | 80.77% | 0.05147 | **85.04%** (Ep 23) | **0.03582** | 23 ep (26.9s) | `gs_acc80.8_h64.json` |
| **Melhor Custo-Beneficio (Mais Rapido no Topo)** | #12 | `H=48, LR=0.008, M=0.85, Val=0.2, P=10, Noise=0.0` | **83.85%** | 0.04972 | 86.32% (Ep 25) | 0.03787 | **31 ep (27.4s)** | `gs_acc83.8_h48.json` |

![Visao Global da Grade](./grid_search_insights.png)

## 2. Sensibilidade por Topologia (Neuronios Ocultos H)

| Neuronios (H) | N | Acc Teste Media (Visual) | Acc Teste Max | MSE Teste Min | Acc Val Media (Max) | MSE Val Min | Gap Val→Teste | Epocas Medias | Tempo Medio (s/ep) |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H=16** | 9 | `████░░░░░░` **66.75%** | **70.77%** | 0.07400 | 76.45% (79.06%) | 0.05571 | +9.70% | 50.3 ep | 15.69s (`0.31s/ep`) |
| **H=48** | 9 | `████████░░` **81.88%** | **83.85%** | 0.04847 | 85.80% (86.75%) | 0.03768 | +3.92% | 43.2 ep | 38.20s (`0.88s/ep`) |
| **H=64** | 9 | `████████░░` **80.00%** | **84.62%** | 0.05147 | 84.76% (85.04%) | 0.03582 | +4.76% | 38.2 ep | 44.80s (`1.17s/ep`) |

## 3. Sensibilidade por Taxa de Aprendizado (Learn Rate)

| Taxa (η) | N | Acc Teste Media (Visual) | Acc Teste Max | MSE Teste Min | Acc Val Media (Max) | MSE Val Min | Gap Val→Teste | Epocas Medias | Tempo Medio (s/ep) |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **η = 0.008** | 9 | `███████░░░` **77.86%** | **84.62%** | 0.04972 | 82.62% (86.32%) | 0.03723 | +4.76% | 52.8 ep | 39.38s (`0.75s/ep`) |
| **η = 0.015** | 9 | `██████░░░░` **75.73%** | **83.85%** | 0.05088 | 82.57% (86.75%) | 0.03582 | +6.85% | 41.1 ep | 31.47s (`0.77s/ep`) |
| **η = 0.025** | 9 | `██████░░░░` **75.04%** | **82.31%** | 0.04847 | 81.81% (86.75%) | 0.03724 | +6.77% | 37.9 ep | 27.83s (`0.73s/ep`) |

## 4. Sensibilidade por Fator de Momentum

| Momentum (α) | N | Acc Teste Media (Visual) | Acc Teste Max | MSE Teste Min | Acc Val Media (Max) | MSE Val Min | Gap Val→Teste | Epocas Medias | Tempo Medio (s/ep) |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **α = 0.0** | 9 | `███████░░░` **77.01%** | **83.85%** | 0.05204 | 82.57% (86.75%) | 0.03856 | +5.57% | 63.7 ep | 47.46s (`0.75s/ep`) |
| **α = 0.55** | 9 | `███████░░░` **77.52%** | **84.62%** | 0.04847 | 83.00% (86.75%) | 0.03724 | +5.48% | 35.8 ep | 27.22s (`0.76s/ep`) |
| **α = 0.85** | 9 | `██████░░░░` **74.10%** | **83.85%** | 0.04972 | 81.43% (86.32%) | 0.03582 | +7.33% | 32.3 ep | 24.00s (`0.74s/ep`) |

![Sensibilidade por Hiperparametro](./grid_search_hyperparams.png)

![Mapas de Calor Bivariados](./grid_search_heatmaps.png)

## 5. Funil de Generalizacao (Treino → Validacao → Teste)

```text
  Treino Final      : [███████████████████░] Media:  98.67% | Max:  99.79%
                           │  (Queda media Treino → Validacao : -16.34 pp)
                           ▼
  Melhor Validacao  : [█████████████░░░░░░░] Media:  82.34% | Max:  86.75%
                           │  (Queda media Validacao → Teste  : -6.12 pp)
                           ▼
  Teste (130 fixos) : [██████████░░░░░░░░░░] Media:  76.21% | Max:  84.62%
```

- **Gap Total Medio (Treino → Teste):** `+22.46%` (Treino→Val: `+16.34%` | Val→Teste: `+6.12%`)
- **Menor Desvio Val→Teste:** Rodada #20 (Val: `84.62%` → Teste: `84.62%` | Gap: `+0.00%`)
- **Menor Sobreajuste Treino→Val:** Rodada #17 (Treino: `99.58%` → Val: `86.75%` | Gap: `+12.83%`)
- **Distribuicao do Sobreajuste (Treino - Validacao):**
  - Baixo (`<= 5%`): **0/27** modelos
  - Moderado (`5% a 15%`): **15/27** modelos
  - Alto (`> 15%`): **12/27** modelos


## 6. Ranking Top 10 Modelos (Ordenado por Acuracia de Teste)

| Pos | Rodada | H | Val | LR | Mom | P | Noise | Acc Teste (Visual) | MSE Teste | Acc Val (Ep) | MSE Val | Acc Treino | Epocas | Tempo (s/ep) | Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **#1** | #20 | 64 | 0.20 | 0.008 | 0.55 | 10 | 0.00 | `███████░` **84.62%** | 0.05621 | 84.62% (Ep 20) | 0.04004 | 99.38% | 30 | 35.0s (`1.17s`) | `gs_acc84.6_h64.json` |
| **#2** | #12 | 48 | 0.20 | 0.008 | 0.85 | 10 | 0.00 | `███████░` **83.85%** | 0.04972 | 86.32% (Ep 25) | 0.03787 | 99.79% | 31 | 27.4s (`0.89s`) | `gs_acc83.8_h48.json` |
| **#3** | #13 | 48 | 0.20 | 0.015 | 0.0 | 10 | 0.00 | `███████░` **83.85%** | 0.05234 | 86.75% (Ep 55) | 0.04042 | 99.79% | 65 | 57.4s (`0.88s`) | `gs_acc83.8_h48_v2.json` |
| **#4** | #11 | 48 | 0.20 | 0.008 | 0.55 | 10 | 0.00 | `███████░` **83.08%** | 0.05162 | 86.32% (Ep 32) | 0.04101 | 99.69% | 42 | 37.1s (`0.88s`) | `gs_acc83.1_h48_v2.json` |
| **#5** | #10 | 48 | 0.20 | 0.008 | 0.0 | 10 | 0.00 | `███████░` **83.08%** | 0.05228 | 85.90% (Ep 69) | 0.04155 | 99.48% | 79 | 69.7s (`0.88s`) | `gs_acc83.1_h48.json` |
| **#6** | #17 | 48 | 0.20 | 0.025 | 0.55 | 10 | 0.00 | `███████░` **82.31%** | 0.04847 | 86.75% (Ep 19) | 0.03788 | 99.58% | 29 | 25.6s (`0.88s`) | `gs_acc82.3_h48_v2.json` |
| **#7** | #14 | 48 | 0.20 | 0.015 | 0.55 | 10 | 0.00 | `███████░` **82.31%** | 0.05088 | 86.75% (Ep 32) | 0.03839 | 99.79% | 42 | 37.2s (`0.89s`) | `gs_acc82.3_h48.json` |
| **#8** | #26 | 64 | 0.20 | 0.025 | 0.55 | 10 | 0.00 | `███████░` **81.54%** | 0.05163 | 85.04% (Ep 32) | 0.03724 | 99.69% | 33 | 38.6s (`1.17s`) | `gs_acc81.5_h64.json` |
| **#9** | #16 | 48 | 0.20 | 0.025 | 0.0 | 10 | 0.00 | `███████░` **81.54%** | 0.05204 | 86.32% (Ep 32) | 0.03909 | 99.69% | 42 | 37.2s (`0.89s`) | `gs_acc81.5_h48.json` |
| **#10** | #24 | 64 | 0.20 | 0.015 | 0.85 | 10 | 0.00 | `██████░░` **80.77%** | 0.05147 | 85.04% (Ep 23) | 0.03582 | 99.58% | 23 | 26.9s (`1.17s`) | `gs_acc80.8_h64.json` |
