# 2. Preenchimento Direto do `docs/Checklist.xlsx`

> **Integrantes do Grupo (EACH-USP | IA 2026 — Entrega 1: MLP):**
> - **Guilherme Oliveira de Souza** — Nº USP: `14671388`
> - **Paulo Rogerio Andrade de Oliveira** — Nº USP: `9365783`
> - **Victor Hugo Antunes Valadares** — Nº USP: `14760317`


## Procedimentos (Linhas 18 a 33 — marcar `x` na Coluna B)
- **Linhas 9 a 14:** Nomes completos e Nº USP dos integrantes.
- **Linha 18:** `x` (Máximo 6 integrantes)
- **Linha 20 ou 21:** `x` (20 = vídeo postado no e-disciplinas | 21 = link externo + PDF de minutagem)
- **Linha 22:** `x` (Ordem do template mantida)
- **Linha 23:** `x` (Todos participam)
- **Linha 24:** `x` (Códigos postados)
- **Linha 25:** `x` (Código documentado com relação à teoria)
- **Linha 26:** `x` (Artefatos com nome e Nº USP)
- **Linha 27 e 28:** `x` (Vídeo entre 5 e 20 min)
- **Linha 31:** `x` (Usou `CARACTERES COMPLETO`)
- **Linha 33:** `x` (Conferência com o Teste de Mesa concluída com sucesso — `lab/mesa_test/mesa_test.py`)

---

## Implementação e Testes MLP (Linhas 36 a 63 — copiar para Colunas B, C, D e E)

| Linha | Quesito (Coluna A) | B | Coluna C (Arquivos) | Coluna D (Linhas) | Coluna E (Extras) |
| :---: | :--- | :---: | :--- | :--- | :--- |
| **36** | Testes com UMA camada escondida | `x` | `src/core/models/mlp.py`<br>`src/core/models/layer.py`<br>`src/main.py` | `mlp.py`: 22–74, 80–93<br>`layer.py`: 20–93<br>`main.py`: 32–59 | Topologia com 1 camada oculta (`hidden_layers=[hidden_layer]`) |
| **37** | Erro minimizado é o MSE | `x` | `src/core/models/mlp.py`<br>`src/core/models/layer.py`<br>`src/training/orchestrator.py` | `mlp.py`: 115–120, 179–186<br>`layer.py`: 99–109<br>`orchestrator.py`: 88–118 | Cálculo de `compute_mse` e derivada $\delta_k = (t_k - y_k)f'(y_{in\_k})$ |
| **38** | Otimização por Gradiente Descendente | `x` | `src/core/models/layer.py`<br>`src/core/models/mlp.py` | `layer.py`: 139–169<br>`mlp.py`: 179–243 | Regra Delta online padrão a padrão ($\Delta w = \alpha \delta z$) |
| **39** | Parada antecipada implementada e testada | `x` | `src/training/orchestrator.py` | `orchestrator.py`: 94–137, 169–180 | Monitoramento do MSE de validação (`patience`) + `target_mse` |
| **42** | Busca de parâmetros | `x` | `src/experiments/grid_search.py`<br>`src/experiments/grid_reporter.py`<br>`lab/analysis/grid_evaluator.py`<br>`lab/analysis/grid_plots.py` | `grid_search.py`: 19–196<br>`grid_reporter.py`: 15–148<br>`grid_evaluator.py`: 54–333<br>`grid_plots.py`: 39–413 | Grid Search variando $H, \eta, \mu$, `val_ratio`, `patience` e `noise_rate` |
| **43** | Variações autorais do conjunto | `x` | `src/data/data_augmentation.py`<br>`src/data/dataset_loader.py` | `data_augmentation.py`: 7–42<br>`dataset_loader.py`: 31–125 | Ruído bipolar (`add_bipolar_noise`) e amostragem estratificada vs. aleatória (`STRATIFIED`) |
| **52** | Outros métodos de otimização | `x` | `src/core/models/layer.py`<br>`src/core/models/mlp.py` | `layer.py`: 52–55, 139–169<br>`mlp.py`: 31, 69, 210–231 | **Gradiente Descendente com Momentum ($\mu$)** |
| **54** | Uso de hold out ou validação cruzada | `x` | `src/data/dataset_loader.py`<br>`src/training/orchestrator.py` | `dataset_loader.py`: 72–125<br>`orchestrator.py`: 94–118 | **Hold-out estratificado** (Treino, Validação e 130 de Teste fixo) |
| **55** | Teste com outros conjuntos de dados | `x` | `lab/logic_gates/logic_gates.py`<br>`lab/mesa_test/mesa_test.py` | `logic_gates.py`: 11–111<br>`mesa_test.py`: 19–120 | Portas lógicas (`AND`, `OR`, `XOR`) e Teste de Mesa oficial |
| **57** | Modelagem de classificação binária | `x` | `lab/logic_gates/logic_gates.py`<br>`src/core/models/mlp.py` | `logic_gates.py`: 13–95<br>`mlp.py`: 122–135 | Saída única ($m=1$) com alvos bipolares $\{-1, +1\}$ |
| **58** | Avaliação de classificação binária | `x` | `src/core/models/mlp.py`<br>`lab/logic_gates/logic_gates.py` | `mlp.py`: 133–148<br>`logic_gates.py`: 104–111 | Limiar de sinal (`>= 0 -> +1.0, < 0 -> -1.0`), MSE e acurácia |
| **62** | *(Extra)* Data Augmentation Bipolar | `x` | `src/data/data_augmentation.py`<br>`src/training/config.py` | `data_augmentation.py`: 7–42<br>`config.py`: 48–75 | Inversão estocástica de pixels ($-1 \leftrightarrow +1$) no treino |
| **63** | *(Extra)* Rollback para Melhores Pesos (`best_weights`) | `x` | `src/training/orchestrator.py`<br>`src/persistence/mlp_persister.py` | `orchestrator.py`: 106–113, 178–180<br>`mlp_persister.py`: 104–177 | Restaura o snapshot da época de menor MSE de validação |
