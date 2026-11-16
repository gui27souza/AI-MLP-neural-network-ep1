# Robustez e Ruído no Treino (`NOISE_RATE`)

- 8 modelos em `H=48, LR=0.025` variando `NOISE_RATE ∈ {0.00, 0.05, 0.15, 0.30}`.
- O ruído reduziu o gap de overfit (`Treino - Val`) de `+13.10%` para `+7.96%`, mantendo **81.54%** no teste com até `15%` de pixels invertidos.
