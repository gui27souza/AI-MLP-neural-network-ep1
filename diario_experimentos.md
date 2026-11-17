# Diário de Experimentos - MLP EP1

- **Grid Search Amplo (192 rodadas):**
  - Mapeamento geral do espaço de hiperparâmetros; identificação de H=48 como topologia ótima (~86.65%).

- **Grid Searches Refinados:**
  - GS 1: Micro-ajuste em H=48 confirmou o campeão (LR=0.010, M=0.55 -> 86.65%).
  - GS 2: Via rápida com alto LR (LR=0.025, M=0.15 -> 86.40% em 19 épocas).
  - GS 3: Exploração do menor MSE em H=64 com alto momentum (MSE 0.03942, mas acc em 84.40%).

- **Avaliação (results/):**
  - Geração de matrizes de confusão, curvas e entregáveis oficiais para o quarteto representativo.

- **Determinismo e Pipeline:**
  - Diagnóstico: falta de semente no shuffle das épocas impedia reproduzir trajetórias ao variar patience.
  - Solução: seed=42 centralizada no Orchestrator e pesos iniciais promovidos para `src/initial_weights.py`.
  - Automação: criação do comando `make full_pipeline` unificando treino determinístico e avaliação.

- **Data Augmentation Bipolar:**
  - Implementado `src/data_augmentation.py` para gerar cópias ruidosas com inversão de pixels (-1 ↔ +1) apenas no treino.
  - Objetivo: avaliar ganho de robustez e generalização contra o baseline limpo sob a mesma semente (seed 42).

- **Grid Search Determinístico (Baseline Limpo - Recorde 87.15%):**
  - Executadas 27 combinações sob determinismo ponta a ponta (seed=42, H ∈ {16, 48, 64}).
  - Novo recorde global: H=48, LR=0.025, M=0.55 -> **87.15%** de acurácia (MSE 0.03918).

- **Grid Search com Data Augmentation (Ruído 3%):**
  - Executadas 27 combinações com 1 cópia aumentada no treino (`NOISE_RATE=0.03`, 1.856 amostras/época).
  - Campeão da rodada: H=48, LR=0.025, M=0.85 -> **85.89%** de acurácia (MSE 0.04437).
  - Resiliência empírica: piso mínimo subiu de 50.38% para 64.99% (+14.6 pp), mitigando colapso sob LR e momentum altos.

- **Refinamento com Paciência Estendida (Recorde Definitivo 87.41%):**
  - Hipótese: patience=6 no grid search encerrava o treino precocemente diante de pequenas oscilações de validação.
  - Teste: ampliação de patience para 15 na melhor configuração limpa (H=48, LR=0.025, M=0.55).
  - Resultado: superação do platô anterior, registrando pico na época 22 com **87.41%** de acurácia de teste e menor MSE global (**0.03852**).
  - Convergência teórica: execução finalizada na época 33 por atingimento do erro alvo formal (`MSE treino = 0.000984 <= 0.001000`).
  - Levei o patience ao limite, coloquei em 50, mas a época 33 de fato foi a melhor, a conclusão desse experimento foi que o modelo de fato tem seu ápice em 33 com acc87.41 e depois começa a perder aos poucos a capacidade de generalização

- Envenenamento do campeão
  - Peguei o modelo campeão (até o momento) de acc87.4 e fiz um grid search em cima das configs dele variando apenas o noise rate com os valores 0.0, 0.03, 0.15, 0.5, 0.9
  - O que aconteceu foi simplesmente uma piora do acc conforme o noise rate aumentou

- Análise da proporção de validação
  - Análise com o melhor modelo (até agora) variando as taxas de proporção de treinamento / validação
  - Nenhuma surpresa, 0.3 para validação seguiu campeão

- Momentum x Learn Rate
  - Análise de valores diferentes de momentum e learn rate no modelo campeão (até agora)

- Novo Grid Search Amplo
  - Grid search buscando analisar o máximo de combinações possíveis entre hiperparâmetros (variando nos extremos do espectro muito baixo, normal, muito alto)
  - LSS - senti na pele o problema de não ter implementado um sistema de salvamento robusto ao longo da execução do grid-search. Fiz uma cirurgia em runtime e consegui recuperar o csv que só seria gerado no final, mas precisei abrir mão das análises com o hiperparâmetro de VAL_RATIO em 0.45
  - Comando que me salvou:
  ```
  sudo gdb -p 5789 -batch \
  -ex 'call (void) PyRun_SimpleString("import gc; [r.save_csv() for r in gc.get_objects() if type(r).__name__ == \"GridSearchReporter\"]")' \
  -ex 'detach' \
  -ex 'quit'
  ```
  > Basciamente pausa o código por um período milimetricamente curto, salva o csv que está <br>
  > apenas em memória acionando o método que só seria acionado ao fim do grid-search completo
  - No quesitos de resultados sobre o melhor modelo, sem novidades
  - No quesito de estudo de relacionamento entre os hiperparâmetros, ficou sensacional

- **Fase 2 — Pipeline Oficial (130 amostras fixas de teste + Early Stopping por MSE):**
  - Hardware: Intel Core i5-14600KF (12+8 @ 5.30 GHz) | 32GB DDR5 6000MT/s.
  - Divisão: 1.196 amostras de desenvolvimento (962 treino / 234 val com `VAL_RATIO=0.20`) e últimas 130 amostras (5 alfabetos, Fontes #47 a #51) isoladas para teste fixo.
  - **1. Grid Search Oficial (`results/1_grid_hiperparametros_oficiais` - 27 modelos):**
    - Campeão Teste: `H=64, LR=0.008, M=0.55` -> **84.62%** no teste (época 20).
    - Campeão H=48 (Momentum): `H=48, LR=0.008, M=0.85` -> **83.85%** teste | **86.32%** val (menor MSE val `0.03787`).
    - Campeão GD Puro: `H=48, LR=0.015, M=0.0` -> **83.85%** teste | **86.75%** val (65 épocas / 57.4s vs ~30 épocas com momentum).
    - Acurácia por fonte nas 130 amostras de teste: Fonte #47 (**96.15%**), Fonte #48 (**69.23%**, estilizada), Fonte #49 (**96.15%**), Fonte #50 (**84.62%**), Fonte #51 (**76.92%**, estilizada).
  - **2. Proporção de Validação (`results/2_grid_proporcao_validacao` - 12 modelos):**
    - `0.20` teve o melhor equilíbrio (**83.85%** teste / **86.64%** val médio), superando `0.10` (val pequena, **81.54%**) e `0.30` (perde amostras de treino, **82.31%** teste).
  - **3. Ruído no Treino (`results/3_grid_robustez_ruido` - 8 modelos):**
    - Reduziu o gap de overfit (`Treino - Val`) de `+13.10%` (limpo) para `+7.96%` (`30%` de ruído), segurando **81.54%** no teste até `15%` de ruído.
  - **4. Comparativo Controlado `H=48, LR=0.025` (`results/4_comparativo_slides_h48_lr025`):**
    - Com Momentum + Early Stopping: **82.31%** teste (`MSE 0.04847`, 29 épocas).
    - GD Puro (`M=0.0`): **81.54%** teste (`MSE 0.04930`, 42 épocas).
    - Sem Early Stopping (60 épocas): decorou o treino (**99.08%**), caiu para **81.54%** no teste (`MSE 0.04928`).
    - Sem Estratificação (`STRATIFIED=false`): despencou para **75.38%** no teste (`-6.93 pp`, `MSE 0.06270`).

- **Fase 3 — Refinamento Final e Cenários da Entrega (`results/fase3_experimentos_finais`):**
  - **1. Funil de 4 Grids (`1_busca_melhor_modelo`):**
    - Descoberta de que `VAL_RATIO=0.30` (832 treino / 364 val) combinado com `PATIENCE=5`, `LR=0.005` e `M=0.85` em `H=48` atinge o **recorde oficial de 85.38% no teste (111/130 acertos)** e **87.09% na validação** em apenas **27 épocas (21.5s)**.
    - O zoom fino (`LR ∈ {0.004, 0.005, 0.006}` x `M ∈ {0.825, 0.85, 0.875}`) confirmou um platô estável com média de **84.19% no teste** e **87.33% na validação**.
  - **2. Quinteto Comparativo Oficial (`2_comparativo_cenarios_finais` — `H=48, LR=0.005`):**
    - Campeão (Momentum `0.85` + Early Stopping `P=5`): **85.38%** teste | **87.09%** val (27 épocas / `21.56s`).
    - GD Puro (`M=0.0` + Early Stopping `P=10`): **83.85%** teste | **88.46%** val (119 épocas / `95.00s` — 4.4x mais lento).
    - Sem Parada Antecipada (`VAL_RATIO=0.0`, 80 épocas): decorou o treino (**99.16%**), caiu para **80.00%** no teste (`-5.38 pp`).
    - Sem Estratificação (`STRATIFIED=false`): caiu para **80.77%** no teste (`-4.61 pp`).
    - Com Ruído 15% (`NOISE_RATE=0.15`): caiu para **76.92%** no teste (`-8.46 pp`).
