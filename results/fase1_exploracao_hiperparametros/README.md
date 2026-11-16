# Fase 1 — Exploração de Hiperparâmetros

- Experimentos iniciais que guiaram o determinismo (`SEED=42`), Data Augmentation e a região ótima (`H ∈ {48, 64}`, `LR ∈ [0.008, 0.025]`).
- Na Fase 2 (raiz de `results/`), o teste passou a isolar estritamente as últimas 130 amostras e o Early Stopping passou a monitorar o MSE de validação.
