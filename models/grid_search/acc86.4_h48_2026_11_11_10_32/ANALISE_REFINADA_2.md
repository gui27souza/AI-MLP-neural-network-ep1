# Relatorio Analitico: Grid Search Refinado 2 (A Via Rapida de Alto LR)

Este documento consolida a analise tecnica e empirica das 18 combinacoes executadas no **Grid Search Refinado 2**, com dados brutos registrados em [summary.csv](file:///Users/guilhermesouza/Desktop/Fauldade-trabalhos/AI-MLP-neural-network-ep1/models/grid_search/acc86.4_h48_2026_11_11_10_32/summary.csv).

---

## 1. Visao Geral da Execucao

- **Total de rodadas:** 18 experimentos focados em convergencia rapida.
- **Tempo total de execucao:** 486,6 segundos (~8,1 minutos).
- **Taxa de Early Stopping:** 18 / 18 (100% de paradas controladas sem divergencia).
- **Melhor acuracia de validacao:** **86,40%** (com MSE de validacao de **0,04139**).
- **Grade avaliada:**
  - `hidden_neurons`: [40, 48, 56]
  - `learn_rate`: [0.015, 0.020, 0.025]
  - `momentum`: [0.05, 0.15]
  - `patience`: [6] (fixo)
  - `val_ratio`: [0.30] (fixo)

![Grafico de Insights Refinado 2](grid_search_refined2_insights.png)

---

## 2. Ranking Geral das 18 Rodadas

| Rank | Rodada | H | LR | Mom | Acc Val (%) | MSE Val | Epocas | Best Ep | Train Acc (%) | Gap (%) | Duracao | Checkpoint |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | **12** | **48** | **0.025** | **0.15** | **86.40** | **0.04139** | **25** | **19** | **99.46** | **13.06** | **25.1s** | [gs_acc86.4_h48.json](file:///Users/guilhermesouza/Desktop/Fauldade-trabalhos/AI-MLP-neural-network-ep1/models/grid_search/acc86.4_h48_2026_11_11_10_32/gs_acc86.4_h48.json) |
| **2** | 10 | 48 | 0.020 | 0.15 | 85.89 | 0.04223 | 36 | 30 | 99.46 | 13.57 | 36.0s | [gs_acc85.9_h48_v3.json](file:///Users/guilhermesouza/Desktop/Fauldade-trabalhos/AI-MLP-neural-network-ep1/models/grid_search/acc86.4_h48_2026_11_11_10_32/gs_acc85.9_h48_v3.json) |
| **3** | 11 | 48 | 0.025 | 0.05 | 85.89 | 0.04236 | 23 | 17 | 99.57 | 13.68 | 24.2s | [gs_acc85.9_h48_v4.json](file:///Users/guilhermesouza/Desktop/Fauldade-trabalhos/AI-MLP-neural-network-ep1/models/grid_search/acc86.4_h48_2026_11_11_10_32/gs_acc85.9_h48_v4.json) |
| **4** | 7 | 48 | 0.015 | 0.05 | 85.89 | 0.04392 | 33 | 22 | 99.46 | 13.57 | 32.5s | [gs_acc85.9_h48.json](file:///Users/guilhermesouza/Desktop/Fauldade-trabalhos/AI-MLP-neural-network-ep1/models/grid_search/acc86.4_h48_2026_11_11_10_32/gs_acc85.9_h48.json) |
| **5** | 9 | 48 | 0.020 | 0.05 | 85.89 | 0.04414 | 26 | 20 | 99.46 | 13.57 | 26.5s | [gs_acc85.9_h48_v2.json](file:///Users/guilhermesouza/Desktop/Fauldade-trabalhos/AI-MLP-neural-network-ep1/models/grid_search/acc86.4_h48_2026_11_11_10_32/gs_acc85.9_h48_v2.json) |
| **6** | 8 | 48 | 0.015 | 0.15 | 84.89 | 0.04445 | 26 | 20 | 99.35 | 14.46 | 27.2s | [gs_acc84.9_h48.json](file:///Users/guilhermesouza/Desktop/Fauldade-trabalhos/AI-MLP-neural-network-ep1/models/grid_search/acc86.4_h48_2026_11_11_10_32/gs_acc84.9_h48.json) |
| **7** | 14 | 56 | 0.015 | 0.15 | 84.38 | 0.04585 | 36 | 30 | 99.57 | 15.19 | 42.4s | [gs_acc84.4_h56.json](file:///Users/guilhermesouza/Desktop/Fauldade-trabalhos/AI-MLP-neural-network-ep1/models/grid_search/acc86.4_h48_2026_11_11_10_32/gs_acc84.4_h56.json) |
| **8** | 16 | 56 | 0.020 | 0.15 | 84.13 | 0.04417 | 23 | 17 | 99.14 | 15.01 | 26.6s | [gs_acc84.1_h56.json](file:///Users/guilhermesouza/Desktop/Fauldade-trabalhos/AI-MLP-neural-network-ep1/models/grid_search/acc86.4_h48_2026_11_11_10_32/gs_acc84.1_h56.json) |
| **9** | 15 | 56 | 0.020 | 0.05 | 83.88 | 0.04510 | 23 | 17 | 99.46 | 15.58 | 27.0s | (descartado < 84%) |
| **10** | 18 | 56 | 0.025 | 0.15 | 83.63 | 0.04334 | 25 | 19 | 99.57 | 15.94 | 30.0s | (descartado < 84%) |
| **11** | 13 | 56 | 0.015 | 0.05 | 83.63 | 0.04493 | 36 | 30 | 99.57 | 15.94 | 41.4s | (descartado < 84%) |
| **12** | 2 | 40 | 0.015 | 0.15 | 83.63 | 0.04551 | 37 | 31 | 99.46 | 15.83 | 30.4s | (descartado < 84%) |
| **13** | 4 | 40 | 0.020 | 0.15 | 83.38 | 0.04507 | 29 | 23 | 99.25 | 15.87 | 24.6s | (descartado < 84%) |
| **14** | 17 | 56 | 0.025 | 0.05 | 83.38 | 0.04548 | 22 | 16 | 99.46 | 16.08 | 25.5s | (descartado < 84%) |
| **15** | 5 | 40 | 0.025 | 0.05 | 82.87 | 0.04586 | 28 | 22 | 99.46 | 16.59 | 23.5s | (descartado < 84%) |
| **16** | 6 | 40 | 0.025 | 0.15 | 82.87 | 0.04868 | 13 | 7 | 98.71 | 15.84 | 10.8s | (descartado < 84%) |
| **17** | 3 | 40 | 0.020 | 0.05 | 82.12 | 0.05013 | 17 | 11 | 98.71 | 16.59 | 14.1s | (descartado < 84%) |
| **18** | 1 | 40 | 0.015 | 0.05 | 81.61 | 0.05161 | 22 | 12 | 98.49 | 16.88 | 18.8s | (descartado < 84%) |

---

## 3. Principais Conclusoes e Fundamentacao Teorica

### 3.1 Consolidacao Cientifica da Capacidade Otima (`H=48`)
O Grid Search 2 fecha uma sequencia empirica irrefutavel sobre a topologia da rede:
- **H = 40:** Média **82,75%** (Max 83,63%). Amargou as ultimas posicoes (12º, 13º, 15º a 18º).
- **H = 48:** Média **85,81%** (Max 86,40%). Monopolizou os 6 primeiros lugares do ranking geral.
- **H = 56:** Média **83,84%** (Max 84,38%).

Em tres baterias independentes de experimentos (Amplo com 192 rodadas, Refinado 1 com 18 rodadas e Refinado 2 com 18 rodadas), todas as configuracoes concorrentes (16, 40, 44, 52, 56, 96, 144) perderam sistematicamente para `H=48`. O ponto de 48 neuronios e matematicamente o sweet spot entre sub-ajuste (falta de capacidade para 26 classes) e sobre-ajuste (dispersao e aumento de variancia).

### 3.2 Eficiencia Extrema da Via Rapida (Rodada 12)
- Hiperparametros: `H=48`, `LR=0.025`, `M=0.15`, `Pat=6`, `ValR=0.30`.
- Desempenho: **86,40%** de acuracia e **0,04139** de MSE de validacao (o menor MSE de validacao entre todas as redes de 48 neuronios avaliadas ate aqui!).
- Tempo de treino: Apenas **19 epocas** e **25,1 segundos** de convergencia.
- Checkpoint gerado: [gs_acc86.4_h48.json](file:///Users/guilhermesouza/Desktop/Fauldade-trabalhos/AI-MLP-neural-network-ep1/models/grid_search/acc86.4_h48_2026_11_11_10_32/gs_acc86.4_h48.json).
- **Licao teorica:** Uma taxa de aprendizado relativamente alta (`0.025`) atua muito bem no treinamento de MLP quando balanceada com Momentum baixo (`0.15`), impedindo o efeito de oscilacao que causou colapsos no grid amplo (onde `LR=0.02` com `M=0.9` quebrou).

### 3.3 Comparativo LR vs Momentum no Regime Rapido
- `LR = 0.025` foi a mais veloz (tempo medio de 23,2s por treino) e gerou o menor MSE.
- `Momentum = 0.15` superou `0.05` em acuracia media (84,36% vs 83,91%) e menor MSE (0,04452 vs 0,04595), fornecendo a inercia minima necessaria para vencer pequenos vales ruidosos sem induzir instabilidade.
