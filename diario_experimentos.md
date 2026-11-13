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

