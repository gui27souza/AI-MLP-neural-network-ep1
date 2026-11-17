# Proporção de Validação (`VAL_RATIO`)

- 12 modelos em `H=48` variando `VAL_RATIO ∈ {0.10, 0.20, 0.30}`.
- `0.20` (962 treino / 234 val) teve o melhor equilíbrio (**83.85%** teste máx / **86.64%** val médio), superando `0.10` (val pequena, **81.54%**) e `0.30` (perde amostras de treino, **82.31%** teste).
