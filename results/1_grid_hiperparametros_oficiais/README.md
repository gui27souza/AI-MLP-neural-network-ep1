# Grid Search Oficial de Hiperparâmetros

- 27 modelos (`H ∈ {16, 48, 64}`, `LR ∈ {0.008, 0.015, 0.025}`, `M ∈ {0.0, 0.55, 0.85}`), `VAL_RATIO=0.20` (962 treino / 234 val / 130 teste fixo).
- `./1_campeao_geral_h64_acc84.6`: Campeão no teste (**84.62%**) com `H=64, LR=0.008, M=0.55`.
- `./2_campeao_h48_momentum_acc83.8`: Campeão H=48 com Momentum (**83.85%** teste / **86.32%** val, menor MSE val `0.03787`).
- `./3_campeao_h48_gd_puro_acc83.8`: Campeão GD Puro `M=0.0` (**83.85%** teste / **86.75%** val, 65 épocas vs ~30 com momentum).
