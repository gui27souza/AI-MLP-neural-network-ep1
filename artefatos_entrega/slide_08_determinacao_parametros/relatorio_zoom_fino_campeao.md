# Relatorio Analitico e Visual — Grid Search

- **Total de Modelos Treinados:** `9` combinacoes
- **Esforco Computacional Total:** `296` epocas acumuladas em `235.5s` (`3.9 min`)
- **Conjunto de Avaliacao Final:** `130 amostras fixas de Teste` (ultimos 5 alfabetos completos, indices `1196..1325`)


## 1. Destaques da Execucao (Campeoes por Criterio)

| Criterio | Rodada | Configuracao | Acc Teste (130 fixos) | MSE Teste | Acc Val (Melhor Ep) | MSE Val | Epocas / Tempo | Checkpoint |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Campeao no Teste (Maior Acc)** | #2 | `H=48, LR=0.004, M=0.85, Val=0.3, P=5, Noise=0.0` | **85.38%** | **0.05392** | 87.91% (Ep 33) | 0.03412 | 38 ep (30.4s) | `gs_acc85.4_h48.json` |
| **Campeao na Validacao (Menor MSE)** | #8 | `H=48, LR=0.006, M=0.85, Val=0.3, P=5, Noise=0.0` | 83.08% | 0.05514 | **86.81%** (Ep 32) | **0.03340** | 37 ep (29.4s) | `-` |
| **Melhor Custo-Beneficio (Mais Rapido no Topo)** | #5 | `H=48, LR=0.005, M=0.85, Val=0.3, P=5, Noise=0.0` | **85.38%** | 0.05241 | 87.09% (Ep 22) | 0.03451 | **27 ep (21.5s)** | `gs_acc85.4_h48_v2.json` |

![Visao Global da Grade](./grid_search_insights.png)

## 2. Sensibilidade por Taxa de Aprendizado (Learn Rate)

| Taxa (η) | N | Acc Teste Media (Visual) | Acc Teste Max | MSE Teste Min | Acc Val Media (Max) | MSE Val Min | Gap Val→Teste | Epocas Medias | Tempo Medio (s/ep) |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **η = 0.004** | 3 | `██████░░░░` **83.85%** | **85.38%** | 0.05392 | 87.64% (87.91%) | 0.03390 | +3.79% | 37.7 ep | 29.97s (`0.80s/ep`) |
| **η = 0.005** | 3 | `████████░░` **84.87%** | **85.38%** | 0.05241 | 87.09% (87.64%) | 0.03360 | +2.22% | 34.0 ep | 27.03s (`0.80s/ep`) |
| **η = 0.006** | 3 | `██████░░░░` **83.85%** | **85.38%** | 0.05246 | 87.27% (87.91%) | 0.03340 | +3.42% | 27.0 ep | 21.48s (`0.80s/ep`) |

## 3. Sensibilidade por Fator de Momentum

| Momentum (α) | N | Acc Teste Media (Visual) | Acc Teste Max | MSE Teste Min | Acc Val Media (Max) | MSE Val Min | Gap Val→Teste | Epocas Medias | Tempo Medio (s/ep) |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **α = 0.825** | 3 | `███████░░░` **84.62%** | **85.38%** | 0.05246 | 87.46% (87.64%) | 0.03400 | +2.84% | 34.3 ep | 27.26s (`0.79s/ep`) |
| **α = 0.85** | 3 | `███████░░░` **84.61%** | **85.38%** | 0.05241 | 87.27% (87.91%) | 0.03340 | +2.66% | 34.0 ep | 27.10s (`0.80s/ep`) |
| **α = 0.875** | 3 | `██████░░░░` **83.34%** | **84.62%** | 0.05384 | 87.27% (87.91%) | 0.03360 | +3.93% | 30.3 ep | 24.13s (`0.80s/ep`) |

![Sensibilidade por Hiperparametro](./grid_search_hyperparams.png)

![Mapas de Calor Bivariados](./grid_search_heatmaps.png)

## 4. Funil de Generalizacao (Treino → Validacao → Teste)

```text
  Treino Final      : [████████████████████] Media:  99.60% | Max:  99.76%
                           │  (Queda media Treino → Validacao : -12.27 pp)
                           ▼
  Melhor Validacao  : [███████████████░░░░░] Media:  87.33% | Max:  87.91%
                           │  (Queda media Validacao → Teste  : -3.14 pp)
                           ▼
  Teste (130 fixos) : [██████████████░░░░░░] Media:  84.19% | Max:  85.38%
```

- **Gap Total Medio (Treino → Teste):** `+15.41%` (Treino→Val: `+12.27%` | Val→Teste: `+3.14%`)
- **Menor Desvio Val→Teste:** Rodada #5 (Val: `87.09%` → Teste: `85.38%` | Gap: `+1.71%`)
- **Menor Sobreajuste Treino→Val:** Rodada #9 (Treino: `98.80%` → Val: `87.91%` | Gap: `+10.89%`)
- **Distribuicao do Sobreajuste (Treino - Validacao):**
  - Baixo (`<= 5%`): **0/9** modelos
  - Moderado (`5% a 15%`): **9/9** modelos
  - Alto (`> 15%`): **0/9** modelos


## 5. Ranking Top 9 Modelos (Ordenado por Acuracia de Teste)

| Pos | Rodada | H | Val | LR | Mom | P | Noise | Acc Teste (Visual) | MSE Teste | Acc Val (Ep) | MSE Val | Acc Treino | Epocas | Tempo (s/ep) | Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **#1** | #5 | 48 | 0.30 | 0.005 | 0.85 | 5 | 0.00 | `██████░░` **85.38%** | 0.05241 | 87.09% (Ep 22) | 0.03451 | 99.52% | 27 | 21.5s (`0.80s`) | `gs_acc85.4_h48_v2.json` |
| **#2** | #7 | 48 | 0.30 | 0.006 | 0.825 | 5 | 0.00 | `██████░░` **85.38%** | 0.05246 | 87.09% (Ep 22) | 0.03440 | 99.52% | 27 | 21.5s (`0.80s`) | `gs_acc85.4_h48_v3.json` |
| **#3** | #2 | 48 | 0.30 | 0.004 | 0.85 | 5 | 0.00 | `██████░░` **85.38%** | 0.05392 | 87.91% (Ep 33) | 0.03412 | 99.76% | 38 | 30.4s (`0.80s`) | `gs_acc85.4_h48.json` |
| **#4** | #4 | 48 | 0.30 | 0.005 | 0.825 | 5 | 0.00 | `██████░░` **84.62%** | 0.05374 | 87.64% (Ep 33) | 0.03400 | 99.76% | 38 | 30.1s (`0.79s`) | `-` |
| **#5** | #6 | 48 | 0.30 | 0.005 | 0.875 | 5 | 0.00 | `██████░░` **84.62%** | 0.05445 | 86.54% (Ep 32) | 0.03360 | 99.76% | 37 | 29.5s (`0.80s`) | `-` |
| **#6** | #1 | 48 | 0.30 | 0.004 | 0.825 | 5 | 0.00 | `█████░░░` **83.85%** | 0.05462 | 87.64% (Ep 33) | 0.03453 | 99.76% | 38 | 30.1s (`0.79s`) | `-` |
| **#7** | #9 | 48 | 0.30 | 0.006 | 0.875 | 5 | 0.00 | `█████░░░` **83.08%** | 0.05384 | 87.91% (Ep 12) | 0.03551 | 98.80% | 17 | 13.5s (`0.80s`) | `-` |
| **#8** | #8 | 48 | 0.30 | 0.006 | 0.85 | 5 | 0.00 | `█████░░░` **83.08%** | 0.05514 | 86.81% (Ep 32) | 0.03340 | 99.76% | 37 | 29.4s (`0.80s`) | `-` |
| **#9** | #3 | 48 | 0.30 | 0.004 | 0.875 | 5 | 0.00 | `████░░░░` **82.31%** | 0.05504 | 87.36% (Ep 32) | 0.03390 | 99.76% | 37 | 29.4s (`0.80s`) | `-` |
