# Plano de Organização e Refatoração — EP1 IA (MLP)

> **Data limite oficial da entrega:** 25 de novembro até 23h55 (via e-Disciplinas)
> **Foco:** Entrega 1 — Multilayer Perceptron (MLP) no dataset `CARACTERES COMPLETO`.

---

## 1. Diagnóstico e Decisões Tomadas

### O que já está excelente e validado
- **Matemática do Backpropagation e Gradiente Descendente (`src/core/models/`):**
  - Testado contra o documento oficial da professora (`docs/exemplo-numerico-MLP.pdf`) — bateu **100% até a 4ª casa decimal** tanto no *feedforward* ($z$ e $y$) quanto no *backward* ($\Delta W, \Delta V$ e pesos atualizados).
- **Infraestrutura do Projeto:**
  - Persistência de checkpoints em JSON e exportação das matrizes $V$ e $W$ no formato exato de Fausett (`coluna 0 = bias`).
  - Geração automática dos 5 arquivos oficiais exigidos na especificação (`hiperparametros.txt`, `pesos_iniciais.csv`, `pesos_finais.csv`, `historico_erro_treinamento.csv`, `saidas_teste_*.csv`).
  - Controle de determinismo (`seed=42`), scripts de Grid Search e plotagem de curvas e matriz de confusão.

### Os 3 Ajustes Críticos (Os "Elefantes na Sala")
1. **Modularização do Core (`MLP` e `Layer`) para os Slides e Checklist:**
   - **Problema atual:** Métodos como `MLP._update_weights` e `Layer.backward` concentram cálculo de erro ($\delta$) e atualização de pesos ($\Delta w, \Delta v$) juntos.
   - **Decisão:** Quebrar em métodos pequenos, atômicos e bem nomeados (~10–15 linhas cada) que espelhem **1-para-1** os Slides 3 a 7 do template da professora e facilitem a indicação de linhas exatas no `docs/Checklist.xlsx`.
2. **Particionamento Formal: Treino / Validação / Teste (130 fixos):**
   - **Problema atual:** O `DatasetLoader` dividia as 1.326 amostras em apenas 2 partes (`train` e `test` via `val_ratio`), usando o mesmo subconjunto para validação (Early Stopping) e teste final.
   - **Decisão:** Seguir estritamente o `data/CARACTERES COMPLETO/README.md`:
     - Total: **1.326 amostras** (51 alfabetos completos de 26 letras `A..Z`).
     - **Teste (Fixo):** Os **últimos 130 exemplos** (índices `1196:1326` = últimos 5 alfabetos completos) ficam isolados exclusivamente para o teste final.
     - **Treino + Validação:** Os **primeiros 1.196 exemplos** (46 alfabetos completos) são divididos entre Treino e Validação conforme `VAL_RATIO` (ou 100% Treino quando `VAL_RATIO=0`, para o experimento sem parada antecipada).
3. **Critério de Parada Antecipada baseado em MSE (e não Acurácia):**
   - **Problema atual:** O `Orchestrator._evaluate_val` salvava o melhor modelo e zerava a paciência priorizando maior acurácia (`acc > best_val_acc`).
   - **Decisão:** Priorizar a redução do **Erro Quadrático Médio de Validação (`val_mse < best_val_mse`)**, conforme exigido no Slide 11 (*"Lembre-se que o erro que está sendo minimizado é o erro quadrático médio na saída da rede neural"*).

---

## 2. Roteiro de Execução em Etapas

### Etapa 1 — Refatoração Modular do Core + Script de Teste de Mesa (CONCLUÍDA ✅)
- [x] Refatorar `src/core/models/layer.py` separando:
  - Cálculo do potencial de ativação (`compute_net_input`, linhas 62–70) e propagação direta (`forward`, linhas 72–93).
  - Cálculo do erro na camada de saída (`compute_output_deltas`, linhas 99–109 — Slide 4).
  - Cálculo do erro retropropagado na camada escondida (`compute_hidden_deltas`, linhas 111–133 — Slide 5).
  - Atualização dos pesos sinápticos e bias (`update_weights`, linhas 139–169 — Slide 6).
- [x] Refatorar `src/core/models/mlp.py` separando:
  - Propagação direta (`predict`, linhas 80–93 — Slide 3) e resposta de reconhecimento de caractere (`predict_class` / `recognize_character`, linhas 95–112 — Slide 7).
  - Cálculo explícito do MSE (`compute_mse`, linhas 115–120 — Slide 3) e da Acurácia (`compute_accuracy`, linhas 122–135).
  - Cálculo do erro na camada de saída (`_compute_output_deltas`, linhas 179–186 — Slide 4).
  - Cálculo da contribuição do erro na camada escondida (`_compute_hidden_deltas`, linhas 188–208 — Slide 5).
  - Aplicação das atualizações de pesos (`_apply_weight_updates`, linhas 210–231 — Slide 6) e orquestração em `_update_weights` (linhas 233–243).
- [x] Criar script `lab/mesa_test/mesa_test.py` reproduzindo o exemplo numérico oficial (`docs/exemplo-numerico-MLP.pdf`) e adicionar alvo no `Makefile` (`make lab_mesa_test`) para comprovar a linha 33 do `Checklist.xlsx`.

### Etapa 2 — Divisão Treino/Validação/Teste (130 fixos) e Early Stopping por MSE (CONCLUÍDA ✅)
- [x] Atualizar `src/data/dataset_loader.py` para retornar `(train_x, train_y, val_x, val_y, test_x, test_y, class_labels)`:
  - Isolar sempre `all_inputs[-130:]` e `all_targets[-130:]` como conjunto de teste fixo (últimos 5 alfabetos).
  - Particionar os `1.196` exemplos iniciais (46 alfabetos) em Treino e Validação segundo `val_ratio` (alinhado em múltiplos de 26 letras com `stratified=True` por padrão, e suporte a `STRATIFIED=false` para o Slide 11).
- [x] Atualizar `src/training/orchestrator.py`:
  - Mudar critério `is_better` na validação para `mse < self.best_val_mse` (usando acurácia apenas como desempate).
  - Receber `test_inputs` e `test_targets` para exibir no sumário do terminal o erro e acurácia de Treino, Validação e Teste (requisito do Slide 10).
- [x] Atualizar `src/main.py`, `src/persistence/mlp_persister.py`, `src/experiments/grid_search.py`, `src/experiments/grid_reporter.py`, `lab/evaluation/metrics.py`, `lab/evaluation/reports.py`, `lab/evaluation/evaluator.py`, `lab/evaluation/deliverables.py`, `lab/analysis/grid_evaluator.py`, `README.md` e scripts `.sh` (`run.sh`, `run.example.sh`, `run_grid.sh`, `run_grid.example.sh`) para refletir os 3 conjuntos (`Treino`, `Validação` e `Teste = 130`).

### Etapa 3 — Reexecução dos Experimentos Chave e Preparação dos Slides/Checklist
- [ ] Rodar o Grid Search e o modelo campeão na nova divisão correta (1.196 treino/val + 130 teste, monitorando MSE).
- [ ] Gerar os cenários específicos exigidos nos Slides 8 a 12:
  - Tabela de dinâmica de treinamento (Hardware $\times$ Tempo $\times$ Épocas $\times$ Neurônios / Taxa de Aprendizado — Slide 9).
  - Duas execuções para demonstração de convergência e parada antecipada (Slide 10).
  - Cenário dificultado para parada antecipada (ruído, amostragem sem estratificação ou fusão de classes `D/O`, `I/J` — Slide 11).
  - Matrizes de confusão comparativas (sem parada antecipada vs. com parada antecipada vs. variações de parâmetros — Slide 12).
- [ ] Preencher `docs/Checklist.xlsx` com os nomes dos arquivos e números exatos de linhas após a refatoração.
