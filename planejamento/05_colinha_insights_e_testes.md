# 5. Colinha de Insights & Guia para Novos Testes

> **Integrantes do Grupo (EACH-USP | IA 2026 — Entrega 1: MLP):**
> - **Guilherme Oliveira de Souza** — Nº USP: `14671388`
> - **Paulo Rogerio Andrade de Oliveira** — Nº USP: `9365783`
> - **Victor Hugo Antunes Valadares** — Nº USP: `14760317`


## 1. Os 5 Insights Essenciais do Baseline Inicial

1. **Taxa de Aprendizado ($\eta$) — O "Pulo do Gato" da Fase 2:**
   - Com as 130 amostras finais fixas e Early Stopping guiado por MSE, **taxas menores (`LR=0.008` e `0.015`) generalizaram melhor no Teste (`84.62%` e `83.85%`)** do que `LR=0.025` (`82.31%`).
   - Passos menores descem mais suavemente na superfície do MSE e acertam mais letras nas 2 fontes estilizadas do conjunto de teste (Fontes #48 e #51).
2. **Topologia ($H$):**
   - `H=16`: sofre *underfitting* (`66.8%` média no teste / `70.8%` máx).
   - **`H=48`**: maior estabilidade média na grade (`81.88%` média no teste / `83.85%` máx, `85.80%` média na validação) com custo de `0.88s/época`.
   - **`H=64`**: bateu o pico isolado no teste (**`84.62%`** = 110/130 acertos com `LR=0.008, M=0.55`), mas teve média ligeiramente menor (`80.00%`) e custa `1.17s/época`.
3. **Momentum ($\mu$) vs. GD Puro ($\mu=0.0$):**
   - **GD Puro (`M=0.0`)** empata perto do topo (`83.85%` no teste com `H=48, LR=0.015`), mas exige **65 épocas (~57s)**.
   - **Momentum (`M=0.55`)** chega em **`84.62%`** cortando as épocas quase pela metade (`35.8` épocas na média vs `63.7` do GD Puro = **-43% de tempo**).
   - **Cuidado com `M=0.85`:** Funciona bem com `LR` baixo (`0.008` $\rightarrow$ `83.85%` em 31 épocas), mas com `LR=0.025` oscila e cai para `76.92%`.
4. **Proporção de Validação (`VAL_RATIO`):**
   - **`0.20` (962 treino / 234 val)** é o ponto ótimo: `0.10` deixa a validação pequena/ruidosa (`81.5%` val) e `0.30` tira 130 amostras do treino (teto no teste cai de `83.85%` para `82.31%`).
5. **Raio-X das 5 Fontes do Teste (`1196..1325`):**
   - Cada letra equivale a **0.77%** de acurácia.
   - Fontes #47 e #49 já batem **96.15%** (25/26); Fonte #50 bate **84.62%** (22/26).
   - A diferença entre um modelo de `82%` e um de `85%+` está em quantas letras ele acerta nas duas fontes mais desenhadas: **Fonte #48 (`69.23%`) e Fonte #51 (`76.92%`)**.

---

## 2. O que já está pronto e pode ser reaproveitado 100%

- `results/2_grid_proporcao_validacao/` (`VAL_RATIO`: `0.10 vs 0.20 vs 0.30`).
- `results/3_grid_robustez_ruido/` (`NOISE_RATE`: `0.00, 0.05, 0.15, 0.30`).
- Controles em `results/4_comparativo_slides_h48_lr025/`: `STRATIFIED=false` (`75.38%`) e `Sem Early Stopping` (`VAL_RATIO=0.0`).

---

## 3. Sugestões de Testes para Explorar no Desktop

### A. Micro-Grid Fino na "Zona Campeã" (`H` intermediário $\times$ `LR` baixa/média)
Explorar o intervalo entre `H=40..64` e `LR=0.005..0.012`:
```bash
make full_grid_pipeline \
  MAX_EPOCHS=120 \
  VAL_RATIOS="0.20" \
  HIDDEN_NEURONS="40,48,56,64" \
  LEARN_RATE="0.005,0.008,0.010,0.012" \
  MOMENTUM="0.0,0.55,0.75" \
  PATIENCE="10" \
  NOISE_RATES="0.0" \
  TARGET_MSE=0.001 \
  MIN_ACC_THRESHOLD=80.0 \
  SEED=42
```

### B. Estudo de Paciência (`PATIENCE`) com `LR` baixa
Testar se `PATIENCE` maior (15 ou 25) permite que `LR=0.008` / `0.010` alcance um vale de MSE ainda melhor:
```bash
make full_grid_pipeline \
  MAX_EPOCHS=150 \
  VAL_RATIOS="0.20" \
  HIDDEN_NEURONS="48,64" \
  LEARN_RATE="0.008,0.010" \
  MOMENTUM="0.0,0.55" \
  PATIENCE="6,10,15,25" \
  NOISE_RATES="0.0" \
  TARGET_MSE=0.001 \
  MIN_ACC_THRESHOLD=80.0 \
  SEED=42
```

### C. Quarteto Comparativo Final com a Melhor Configuração Encontrada
Depois de escolher o melhor `H` e `LR` nos testes acima, rodar os 4 cenários isolados (Com Momentum, GD Puro `M=0.0`, Sem Early Stopping `VAL_RATIO=0.0` e Sem Estratificar `STRATIFIED=false`) com esses mesmos valores para fechar os gráficos dos Slides 10, 11 e 12.
