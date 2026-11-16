# Mapeamento: Slides Oficial (`template-slides.pptx`) $\leftrightarrow$ Código

> **Regra de Ouro da Especificação:** A ordem dos slides do template **NÃO** pode ser alterada (é permitido adicionar slides extras entre os tópicos ou ao final, mantendo o vídeo entre 5 e 20 minutos).

---

## Parte 1 — Arquitetura e Algoritmo (Slides 2 a 7)

| Slide | Título no Template | O que deve ser explicado | Método / Trecho Alvo no Código (Pós-Etapa 1) |
| :--- | :--- | :--- | :--- |
| **Slide 2** | **Estruturas de dados e programação** | - Camada de entrada (120 pixels = imagem $10\times 12$)<br>- Camada de saída (26 neurônios = classes `'A'..'Z'` em One-Hot Bipolar `[-1, +1]`)<br>- Camada escondida ($p$ neurônios)<br>- Matrizes de pesos $V_{(p \times 121)}$ e $W_{(26 \times p+1)}$ | - `DatasetLoader.load_caracteres_completo` (`src/data/dataset_loader.py`)<br>- `Layer.__init__` (`src/core/models/layer.py#L20-L56`)<br>- `MLP.__init__` (`src/core/models/mlp.py#L22-L74`) |
| **Slide 3** | **Feedforward e cálculo do erro da rede** | - Propagação direta: $z_{in\_j} = v_{0j} + \sum x_i v_{ij} \Rightarrow z_j = f(z_{in\_j})$ e $y_k = f(y_{in\_k})$<br>- Cálculo do Erro Quadrático Médio (MSE) | - `Layer.compute_net_input` e `Layer.forward` (`src/core/models/layer.py#L62-L93`)<br>- `MLP.predict` e `MLP.compute_mse` (`src/core/models/mlp.py#L80-L120`) |
| **Slide 4** | **Backpropagation – cálculo do erro para retropropagação** | - Termo de erro na camada de saída:<br>$\delta_k = (t_k - y_k) \cdot f'(y_{in\_k})$ | - `Layer.compute_output_deltas` (`src/core/models/layer.py#L99-L109`)<br>- `MLP._compute_output_deltas` (`src/core/models/mlp.py#L179-L186`) |
| **Slide 5** | **Backpropagation – contribuição do erro na camada escondida** | - Soma ponderada dos erros da camada seguinte e multiplicação pela derivada:<br>$\delta_{in\_j} = \sum_k \delta_k w_{jk}$ e $\delta_j = \delta_{in\_j} \cdot f'(z_{in\_j})$ | - `Layer.compute_hidden_deltas` (`src/core/models/layer.py#L111-L133`)<br>- `MLP._compute_hidden_deltas` (`src/core/models/mlp.py#L188-L208`) |
| **Slide 6** *(Seu)* | **Atualização de pesos sinápticos** | - Regra Delta do Gradiente Descendente (e extensão opcional com Momentum):<br>$\Delta w_{jk} = \alpha \delta_k z_j + \mu \Delta w_{jk}(t-1)$<br>$w_{jk} \leftarrow w_{jk} + \Delta w_{jk}$ (e análogo para bias e matriz $V$) | - `Layer.update_weights` (`src/core/models/layer.py#L139-L169`)<br>- `MLP._apply_weight_updates` e `MLP._update_weights` (`src/core/models/mlp.py#L210-L243`) |
| **Slide 7** *(Seu)* | **Resposta de reconhecimento de caractere** | - Como o vetor contínuo de 26 saídas em $(-1, +1)$ é convertido na letra reconhecida:<br>- Regra *Winner-Takes-All* (`argmax` do neurônio com maior ativação) | - `MLP.predict_class` e `MLP.recognize_character` (`src/core/models/mlp.py#L95-L112`) |

---

## Parte 2 — Experimentação, Dinâmica e Parada Antecipada (Slides 8 a 13)

| Slide | Título no Template | O que deve ser apresentado | Ferramenta / Fonte no Repositório |
| :--- | :--- | :--- | :--- |
| **Slide 8** *(Seu)* | **Determinação de parâmetros** | - Como os parâmetros são definidos (via CLI / variáveis de ambiente no `Makefile` / `TrainConfig`)<br>- Quais parâmetros variam (`HIDDEN_NEURONS`, `LEARN_RATE`, `MOMENTUM`, `PATIENCE`, `VAL_RATIO`, `NOISE_RATE`)<br>- Procedimento de busca (Grid Search) com gráficos e tabelas mostrando quais parâmetros mais influenciam | - `src/training/config.py` e `Makefile`<br>- `src/experiments/grid_search.py`<br>- Gráficos de `lab/analysis/grid_plots.py` (`grid_search_insights.png`, `grid_search_hyperparams.png`) |
| **Slide 9** *(Seu)* | **Dinâmica de treinamento e teste** | - Hardware utilizado nos treinamentos<br>- Tabelas relacionando:<br>  1. Neurônios ocultos $\times$ Tempo $\times$ Épocas<br>  2. Taxa de aprendizado $\times$ Tempo $\times$ Épocas<br>  3. Quantidade de épocas $\times$ Tempo | - Colunas `duration_seconds` e `epochs_executed` do `summary.csv` gerado pelo Grid Search |
| **Slide 10** *(Seu)* | **Treinamento e parada antecipada** | - Mostrar pelo menos **2 execuções** (no terminal ou vídeo incorporado)<br>- Destacar convergência do **MSE de treino e de validação**, e mostrar o **erro no conjunto de teste** ao final<br>- Informar parametrização, convergência dos pesos e **qual é o conjunto de treino, validação e teste** | - Saída de console do `Orchestrator.run` e `Orchestrator.print_results`<br>- Relatório `hiperparametros.txt` |
| **Slide 11** *(Seu)* | **Parada antecipada (Gráficos)** | - Gráficos das curvas de MSE (treino vs. validação) destacando o ponto de parada antecipada<br>- Experimentos dificultando o ambiente (injeção de ruído, amostragem aleatória sem estratificação, ou fusão de classes parecidas como `D+O` / `I+J`) | - `learning_curves.png` gerado por `lab/evaluation/plots.py`<br>- Experimentos com `NOISE_RATE` / amostragem |
| **Slide 12** | **Matriz de confusão** | - Plotagem de matrizes de confusão sob diferentes condições:<br>  1. Variação de parâmetros<br>  2. Treinamento completo **sem** parada antecipada<br>  3. Treinamento **com** parada antecipada | - `confusion_matrix.png` (`lab/evaluation/plots.py`)<br>- Análise de maiores confusões do `evaluation_report.txt` (`lab/evaluation/reports.py`) |
| **Slide 13** | **Implementações adicionais** | - Apresentar implementações extras se sobrar tempo dentro dos 20 min (Hold-out, Momentum, Data Augmentation Bipolar, etc.) | - Ver detalhamento em `02_features_adicionais_e_checklist.md` |
