# Features Adicionais, Opcionais e Checklist (`docs/Checklist.xlsx`)

Este documento separa claramente:
1. O que são **implementações adicionais de Rede Neural / Treinamento** (para o **Slide 13** e linhas 51–61 do `Checklist.xlsx`).
2. O que são **ferramentas de análise e avaliação** (para enriquecer os **Slides 8 a 12**).
3. O que fica no **Backlog (`TODO.md`)** apenas se sobrar tempo.

---

## 1. Implementações Adicionais JÁ PRESENTES no Repositório (Slide 13 + Checklist)

Na planilha `docs/Checklist.xlsx` (seção **"Implementações a mais – opcionais"**, linhas 51 a 61), a professora lista quesitos extras. Veja o que o projeto **já tem pronto**:

| Quesito no `Checklist.xlsx` | O que temos implementado | Onde está no código |
| :--- | :--- | :--- |
| **Linha 52:** *Outros métodos de otimização para MLP. Informe quais:* | **Gradiente Descendente com Momentum ($\mu$)**<br>Adiciona inércia à atualização de pesos ($\Delta w(t) = \alpha \delta z + \mu \Delta w(t-1)$) para acelerar convergência em platôs e amortecer oscilações. | `src/core/models/mlp.py`<br>`src/core/models/layer.py` |
| **Linha 54:** *Uso de hold out ou validação cruzada no MLP.* | **Estratégia Hold-out Parametrizável**<br>Separação formal da base de desenvolvimento em Treino e Validação (`VAL_RATIO`) isolada do conjunto de Teste fixo (130 amostras). | `src/data/dataset_loader.py`<br>`src/training/orchestrator.py` |
| **Linha 43 / Linha 61 (Extras):** *Variações (autorais) do conjunto de dados / Complementações* | **Data Augmentation Bipolar (Injeção de Ruído Estocástico)**<br>Geração sintética de amostras com inversão controlada de pixels ($-1 \leftrightarrow +1$) parametrizada por `NOISE_RATE` e `AUGMENT_COPIES`. | `src/data/data_augmentation.py` |
| **Linhas 57 e 58:** *Modelagem e avaliação de problema de classificação binário no MLP* | **Suporte Nativo a Classificação Binária Bipolar**<br>A arquitetura suporta saída única ($m=1$ com decisão por limiar de sinal $\ge 0 \rightarrow +1$, $< 0 \rightarrow -1$), validada nas portas lógicas (`AND`, `OR`, `XOR`). | `src/core/models/mlp.py` (`evaluate`)<br>`lab/logic_gates/logic_gates.py` |
| **Linha 61 (Extras):** *Complementações adicionais* | **1. Restauração Automática do Melhor Checkpoint (Best-Weights Rollback):** Ao acionar o Early Stopping, a rede restaura em memória os pesos exatos da época de menor erro de validação.<br>**2. Duplo Critério de Parada Antecipada:** Suporte simultâneo a parada por paciência na validação (`PATIENCE`) e por atingimento de erro alvo (`TARGET_MSE`). | `src/training/orchestrator.py`<br>`src/persistence/mlp_persister.py` |

---

## 2. Ferramentas de Análise (Para usar nos Slides 8, 9, 10 e 12 — NÃO como feature de RNA no Slide 13)

Como você bem pontuou com o grupo, o relatório `evaluation_report.txt` **não é uma feature do algoritmo da MLP**, mas sim uma **interpretação analítica da Matriz de Confusão e do Teste**. O lugar ideal para brilhar com esses itens é durante os slides de resultados:

- **Relatório de Maiores Confusões e Acurácia por Classe (`evaluation_report.txt` / `lab/evaluation/reports.py`):**
  - Perfeito para o **Slide 12 (Matriz de Confusão)**! Permite explicar rapidamente quais pares de letras a rede mais confunde (ex.: `D` com `O`, `I` com `J`, `M` com `N`) e quais letras têm 100% de acerto.
- **Motor de Busca em Grade (`grid_search.py` + `grid_evaluator.py` + `grid_plots.py`):**
  - Perfeito para o **Slide 8 (Determinação de parâmetros)** e **Slide 9 (Dinâmica de treinamento e teste)**, pois já gera o CSV com os tempos de execução, épocas e gráficos de impacto de cada hiperparâmetro.

---

## 3. Backlog Opcional (`TODO.md` — Apenas se sobrar tempo)

Itens anotados no `TODO.md` que **não são obrigatórios**, mas que podem ser adicionados depois se quisermos ainda mais conteúdo para o Slide 13:
1. **Inicialização de Nguyen-Widrow:** Escalonamento dos pesos iniciais da camada escondida por $\beta = 0.7 \cdot p^{1/n}$ (Fausett, seção 6.3).
2. **Decaimento da Taxa de Aprendizado (Learning Rate Decay):** Redução gradual de $\eta$ ao longo das épocas.
3. **Eliminação de *Flat Spots*:** Adição de constante $+0.1$ na derivada da sigmoide bipolar.
4. **Regularização L2 (*Weight Decay*):** Penalização proporcional à magnitude dos pesos.
