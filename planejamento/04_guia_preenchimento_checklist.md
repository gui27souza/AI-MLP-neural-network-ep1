# Guia de Preenchimento do `docs/Checklist.xlsx` + Especificação de Hardware (Slide 9)

Este documento traz todas as linhas referentes à **Entrega 1 (MLP)** da planilha `docs/Checklist.xlsx` já mapeadas com os nomes exatos dos arquivos e números de linhas do repositório refatorado, além dos dados de hardware da sua máquina para o **Slide 9**.

---

## 1. Especificação de Hardware e Ambiente (Para o Slide 9 — "Dinâmica de treinamento e teste")

No **Slide 9**, o primeiro item obrigatório é informar o hardware utilizado nos experimentos:
- **Modelo / Processador (SoC):** Apple M1 Pro (Arquitetura ARM64, 8 núcleos de CPU)
- **Memória RAM:** 16 GB de Memória Unificada
- **Sistema Operacional:** macOS
- **Ambiente de Execução:** Python `3.13.18` (execução 100% em CPU single-process, utilizando listas nativas no core da MLP e `numpy` apenas para operações auxiliares de I/O e métricas).

---

## 2. Mapeamento Linha a Linha do `docs/Checklist.xlsx` (Foco MLP)

> **Nota da planilha:** Nas linhas 18 a 33 (*Procedimentos*), as colunas C, D e E são cinzas (só se preenche a coluna **B** com `x`). Nas linhas 36 em diante, preenchem-se as colunas **B (`x`)**, **C (`Código - nome dos arquivos`)**, **D (`Linhas do arquivo`)** e **E (`Extras`)**.

### Seção: Procedimentos (Linhas 18 a 33)
| Linha | Quesito | Check (B) | Observação |
| :---: | :--- | :---: | :--- |
| **18** | O grupo tem no máximo seis integrantes | `x` | Preencher nomes e Nº USP nas linhas 9 a 14 |
| **20 / 21** | Os dois vídeos foram postados no e-disciplinas (20) ou em outro repositório com PDF de minutagem (21) | `x` | Marcar a linha 20 ou 21 conforme forma de envio |
| **22** | Os vídeos foram gravados seguindo rigorosamente o template de slides | `x` | Ordem dos slides mantida integralmente |
| **23** | Todos os integrantes do grupo aparecem em pelo menos um dos vídeos com participação significativa | `x` | — |
| **24** | Os arquivos de código foram postados no e-disciplinas | `x` | — |
| **25** | O código está muito bem documentado, fazendo relação com a teoria da disciplina | `x` | Comentários e docstrings seguem nomenclatura de Fausett |
| **26** | Todos os artefatos postados possuem o nome completo e o número USP de todos os integrantes | `x` | Incluir cabeçalho/identificação nos artefatos finais |
| **27** | O vídeo de MLP tem no mínimo 5 minutos | `x` | — |
| **28** | O vídeo de MLP tem no máximo 20 minutos | `x` | — |
| **31** | O conjunto CARACTERES COMPLETO foi usado para explicações no vídeo de MLP | `x` | — |
| **33** | Foi feita uma conferência dos resultados de nosso código com o exposto no arquivo “Apoio a testes de mesa (MLP) Arquivo” e foi obtido sucesso. | `x` | Validado em `lab/mesa_test/mesa_test.py` (`make lab_mesa_test`) |

---

### Seção: Implementação — Obrigatórios MLP (Linhas 36 a 39)

| Linha | Quesito (Coluna A) | Check (B) | Código — nome dos arquivos (Coluna C) | Linhas do arquivo (Coluna D) | Extras / Detalhes (Coluna E) |
| :---: | :--- | :---: | :--- | :--- | :--- |
| **36** | A implementação do MLP conta com testes realizados com UMA camada escondida | `x` | `src/core/models/mlp.py`<br>`src/core/models/layer.py`<br>`src/main.py` | `mlp.py`: linhas 22–74, 80–93<br>`layer.py`: linhas 20–93<br>`main.py`: linhas 32–59 | Topologia com 1 camada escondida (`hidden_layers=[hidden_layer]`) |
| **37** | O erro minimizado no processo de treinamento do MLP é o Erro Quadrático Médio | `x` | `src/core/models/mlp.py`<br>`src/core/models/layer.py`<br>`src/training/orchestrator.py` | `mlp.py`: linhas 115–120, 179–186<br>`layer.py`: linhas 99–109<br>`orchestrator.py`: linhas 88–118 | Cálculo explícito do MSE (`compute_mse`) e derivada do erro quadrático $\delta_k = (t_k - y_k)f'(y_{in\_k})$ |
| **38** | O método de otimização do MLP é o Gradiente Descendente | `x` | `src/core/models/layer.py`<br>`src/core/models/mlp.py` | `layer.py`: linhas 139–169<br>`mlp.py`: linhas 179–243 | Regra Delta online padrão a padrão ($\Delta w = \alpha \delta z$) |
| **39** | A parada antecipada foi implementada e testada no MLP | `x` | `src/training/orchestrator.py` | `orchestrator.py`: linhas 94–137, 169–180 | Monitoramento do MSE de validação (`patience`) com restauração dos melhores pesos (`best_weights`) e suporte a `target_mse` |

---

### Seção: Testes – Exploração da MLP (Linhas 42 a 43)

| Linha | Quesito (Coluna A) | Check (B) | Código — nome dos arquivos (Coluna C) | Linhas do arquivo (Coluna D) | Extras / Detalhes (Coluna E) |
| :---: | :--- | :---: | :--- | :--- | :--- |
| **42** | Foi feita uma busca de parâmetros | `x` | `src/experiments/grid_search.py`<br>`src/experiments/grid_reporter.py`<br>`lab/analysis/grid_evaluator.py`<br>`lab/analysis/grid_plots.py` | `grid_search.py`: linhas 19–196<br>`grid_reporter.py`: linhas 15–148<br>`grid_evaluator.py`: linhas 38–213<br>`grid_plots.py`: linhas 32–309 | Busca em grade automatizada variando $H$, $\eta$, $\mu$, `val_ratio`, `patience` e `noise_rate` |
| **43** | Os testes contaram com variações (autorais) do conjunto de teste | `x` | `src/data/data_augmentation.py`<br>`src/data/dataset_loader.py` | `data_augmentation.py`: linhas 7–42<br>`dataset_loader.py`: linhas 31–125 | Injeção paramétrica de ruído bipolar (`add_bipolar_noise`) e variação de amostragem estratificada vs. aleatória (`STRATIFIED`) |

---

### Seção: Implementações a mais – Opcionais MLP (Linhas 52 a 62)

| Linha | Quesito (Coluna A) | Check (B) | Código — nome dos arquivos (Coluna C) | Linhas do arquivo (Coluna D) | Extras / Detalhes (Coluna E) |
| :---: | :--- | :---: | :--- | :--- | :--- |
| **52** | Outros métodos de otimização para MLP. Informe quais: | `x` | `src/core/models/layer.py`<br>`src/core/models/mlp.py` | `layer.py`: linhas 52–55, 139–169<br>`mlp.py`: linhas 31, 69, 210–231 | **Gradiente Descendente com Momentum ($\mu$):** adiciona termo inercial $\mu \cdot \Delta w(t-1)$ |
| **54** | Uso de hold out ou validação cruzada no MLP. | `x` | `src/data/dataset_loader.py`<br>`src/training/orchestrator.py` | `dataset_loader.py`: linhas 72–125<br>`orchestrator.py`: linhas 94–118 | **Hold-out estratificado (e aleatório):** separação formal em Treino, Validação (`val_ratio`) e Teste fixo (130 amostras) |
| **55** | Teste com outros conjuntos de dados no MLP. | `x` | `lab/logic_gates/logic_gates.py`<br>`lab/mesa_test/mesa_test.py` | `logic_gates.py`: linhas 11–111<br>`mesa_test.py`: linhas 19–120 | Testes com portas lógicas (`AND`, `OR`, `XOR`) e conjunto do Teste de Mesa oficial |
| **57** | Modelagem de um problema de classificação binário no MLP. | `x` | `lab/logic_gates/logic_gates.py`<br>`src/core/models/mlp.py` | `logic_gates.py`: linhas 13–95<br>`mlp.py`: linhas 122–135 | Modelagem bipolar com 1 neurônio de saída ($m=1$, alvos em $\{-1, +1\}$) |
| **58** | Avaliação de um problema de classificação binário no MLP | `x` | `src/core/models/mlp.py`<br>`lab/logic_gates/logic_gates.py` | `mlp.py`: linhas 133–135, 137–148<br>`logic_gates.py`: linhas 104–111 | Decisão por limiar de sinal (`outputs >= 0 -> +1.0, < 0 -> -1.0`) e cálculo de MSE e acurácia |
| **62** *(Nova linha)* | Data Augmentation Bipolar (Injeção Estocástica de Ruído em Pixels) | `x` | `src/data/data_augmentation.py`<br>`src/training/config.py` | `data_augmentation.py`: linhas 7–42<br>`config.py`: linhas 48–75 | Geração de cópias sintéticas com inversão controlada de pixels ($-1 \leftrightarrow +1$) |
| **63** *(Nova linha)* | Rollback Automático para os Melhores Pesos de Validação (Best-Weights Checkpoint) | `x` | `src/training/orchestrator.py`<br>`src/persistence/mlp_persister.py` | `orchestrator.py`: linhas 106–113, 178–180<br>`mlp_persister.py`: linhas 104–177 | Restaura em memória o snapshot exato da época de menor MSE de validação ao encerrar o treino |
