# 2. Comparativo dos 5 Cenários Finais (`H=48, LR=0.005, VAL_RATIO=0.30`)

Isolando uma variável por vez a partir do modelo campeão (`SEED=42`):

1. `./1_campeao_oficial_momentum_es_acc85.4` (`M=0.85, P=5, STRATIFIED=true`): **85.38%** no teste (`MSE 0.05241`), **87.09%** na val (`MSE 0.03451`), parou na época 27 (melhor: **22**, `21.56s`).
2. `./2_gd_puro_sem_momentum_es_acc83.8` (`M=0.00, P=10`): **83.85%** no teste (`MSE 0.05582`), **88.46%** na val, exigiu **119 épocas** (melhor: **109**, `95.00s` — 4.4x mais lento).
3. `./3_sem_parada_antecipada_80ep_acc80.0` (`VAL_RATIO=0.0`, 80 épocas fixas): decorou o treino (**99.16%**), mas caiu para **80.00%** no teste (`-5.38 pp`, `84.17s`).
4. `./4_sem_estratificacao_aleatorio_acc80.8` (`STRATIFIED=false`): caiu para **80.77%** no teste (`-4.61 pp`, `MSE 0.05444`).
5. `./5_com_ruido_15pct_acc76.9` (`NOISE_RATE=0.15`): caiu para **76.92%** no teste (`-8.46 pp`, `MSE 0.05753`).
