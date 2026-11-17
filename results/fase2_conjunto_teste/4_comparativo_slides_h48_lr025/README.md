# Comparativo Controlado (`H=48, LR=0.025`)

- `./1_com_momentum_e_early_stopping_acc82.3`: Baseline (`M=0.55`), parou na época 29 (melhor: 19) -> **82.31%** teste (`MSE 0.04847`).
- `./2_gd_puro_sem_momentum_acc81.5`: Sem momentum (`M=0.0`), levou 42 épocas (melhor: 32) -> **81.54%** teste (`MSE 0.04930`).
- `./3_sem_early_stopping_60ep_acc81.5`: 60 épocas fixas (`VAL_RATIO=0.0`), decorou o treino (**99.08%**) e caiu para **81.54%** no teste (`MSE 0.04928`).
- `./4_sem_estratificacao_aleatorio_acc75.4`: Divisão aleatória (`STRATIFIED=false`), caiu para **75.38%** no teste (`-6.93 pp`, `MSE 0.06270`).
