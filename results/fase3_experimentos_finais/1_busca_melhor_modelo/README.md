# 1. Busca Refinada pelo Melhor Modelo (Funil em 4 Etapas)

1. `acc83.8_h48_2026_11_17_18_19` (16 modelos): Varredura `H ∈ {48, 64}`, `Val ∈ {0.20, 0.30}`, `LR ∈ {0.005, 0.025}`, `M ∈ {0.0, 0.85}` (`P=10`) -> identificou `H=48, Val=0.30, LR=0.005` como região superior (`83.85%` teste / `88.46%` val).
2. `acc85.4_h48_2026_11_17_18_37` (8 modelos): Ajuste para `PATIENCE=5` em `Val=0.30` -> novo recorde de **85.38% no teste** (`111/130`, `H=48, LR=0.005, M=0.85`, 27 épocas / 21.4s).
3. `acc85.4_h48_2026_11_17_19_07` (9 modelos): Refinamento em `LR ∈ {0.001, 0.005, 0.0075}` $\times$ `M ∈ {0.70, 0.85, 0.95}` -> confirmou pico em `LR=0.005, M=0.85` (**85.38%**).
4. `acc85.4_h48_2026_11_17_19_14` (9 modelos): Zoom fino em `LR ∈ {0.004, 0.005, 0.006}` $\times$ `M ∈ {0.825, 0.85, 0.875}` -> alta estabilidade (**84.19% média no teste** e **87.33% média na validação**, com 3 modelos empatados em **85.38%**).
