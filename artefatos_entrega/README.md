# Artefatos Selecionados da Entrega 1 (MLP) — Guia Visual

> **Integrantes do Grupo (EACH-USP | IA 2026 — Entrega 1: MLP):**
> - **Guilherme Oliveira de Souza** — Nº USP: `14671388`
> - **Paulo Rogerio Andrade de Oliveira** — Nº USP: `9365783`
> - **Victor Hugo Antunes Valadares** — Nº USP: `14760317`


Esta pasta reúne **apenas os gráficos, relatórios e arquivos de saída finais (cherry-pick)** prontos para inserir nos slides e postar no e-Disciplinas, junto com **notas de leitura** para ninguém ter dúvida na hora de apresentar.

---

## 1. `slide_08_determinacao_parametros/` (Para o Slide 8)

Como usamos a busca em grade em funil para encontrar a configuração campeã (`H=48, LR=0.005, M=0.85, VAL_RATIO=0.30, PATIENCE=5` $\rightarrow$ **85,38% no teste**):

| Arquivo | O que mostra / Como usar no Slide 8 |
| :--- | :--- |
| `1_sensibilidade_geral_H16_48_64_LR_M.png` | Mostra por que $H=16$ sofre *underfitting* (`66,8%` teste) e $H=48$ tem a maior média (`81,9%` teste / `85,8%` val). |
| `2_sensibilidade_proporcao_validacao.png` | Mostra o impacto de variar `VAL_RATIO` (`10%`, `20%`, `30%`). |
| `3_refinamento_patience5_85.4pct.png` | Mostra o salto para **85,38%** ao ajustar `PATIENCE=5` em `H=48` vs `H=64`. |
| `4_heatmap_LR_vs_Momentum_85.4pct.png` | Mapa de calor $\eta \times \mu$ destacando o pico em `LR=0.005` e `M=0.85` (**85,38%**). |
| `5_heatmap_zoom_fino_plato_85.4pct.png` | Zoom fino provando estabilidade: média da grade inteira em **84,19% no teste** e **87,33% na validação**. |
| `relatorio_grid_geral.md` / `relatorio_zoom_fino_campeao.md` | Tabelas completas de suporte com todas as métricas de validação e teste. |

### 1.1 Sensibilidade Geral (`H ∈ {16, 48, 64}`, `LR`, `Momentum`)
> [!NOTE]
> **Como ler:** As barras cheias mostram a **média** de todos os modelos daquela configuração; os marcadores (`◆` azul e `★` laranja) mostram o **pico (melhor modelo)** atingido em Validação e Teste. Note o salto de $H=16$ para $H=48$ (+15 pp de média no teste).

![Sensibilidade Geral](./slide_08_determinacao_parametros/1_sensibilidade_geral_H16_48_64_LR_M.png)

### 1.2 Sensibilidade da Proporção de Validação (`VAL_RATIO`: `10%`, `20%`, `30%`)
> [!NOTE]
> **Por que escolhemos `0.30` (30%)?** Com `10%` (apenas ~4 exemplos por letra na validação), o MSE de validação fica mais sujeito a sorte no sorteio das fontes. Com `30%` (**364 exemplos de validação = 14 por letra**), a validação atinge a maior média (`85,9%`) e estabiliza o gatilho da parada antecipada.

![Proporção de Validação](./slide_08_determinacao_parametros/2_sensibilidade_proporcao_validacao.png)

### 1.3 Refinamento com `PATIENCE=5` e `VAL_RATIO=0.30` (Salto para **85,38%**)
![Refinamento Patience 5](./slide_08_determinacao_parametros/3_refinamento_patience5_85.4pct.png)

### 1.4 Interação Taxa de Aprendizado ($\eta$) $\times$ Momentum ($\mu$)
![Heatmap LR vs Momentum](./slide_08_determinacao_parametros/4_heatmap_LR_vs_Momentum_85.4pct.png)

### 1.5 Zoom Fino no Platô Campeão (`LR ∈ {0.004, 0.005, 0.006}` $\times$ `M ∈ {0.825, 0.85, 0.875}`)
> [!NOTE]
> **Por que este gráfico tem só 1 painel em vez de 3?** Porque no Grid 4 (zoom fino) fixamos `H=48`, `VAL_RATIO=0.30` e `PATIENCE=5` e variamos apenas $\eta \times \mu$. Todos os 9 modelos da grade ficaram entre **83,1% e 85,4% no teste** (média `84,19%`), provando que `85,38%` é um platô estável e não sorte de semente.

![Heatmap Zoom Fino](./slide_08_determinacao_parametros/5_heatmap_zoom_fino_plato_85.4pct.png)

---

## 2. `slide_09_dinamica_e_custo/` (Para o Slide 9)

Relação entre topologia ($H$), taxa de aprendizado ($\eta$), momentum ($\mu$), épocas e tempo (em `Intel Core i5-14600KF | 32GB DDR5`):

| Arquivo | O que mostra / Como usar no Slide 9 |
| :--- | :--- |
| `1_custo_computacional_H16_H48_H64.png` | Gráfico de bolhas (Tempo vs Acurácia de Teste vs Épocas) comparando $H=16$ (`0,31s/ép`), $H=48$ (`0,79–0,88s/ép`) e $H=64$ (`1,05–1,17s/ép`). |
| `2_custo_e_funil_campeao_85.4pct.png` | Fronteira MSE, Funil `Treino → Val → Teste` e custo do campeão (**27 épocas em 21,5s**). |

### 2.1 Custo Computacional e Funil (`H=16` vs `H=48` vs `H=64`)
![Custo H16 H48 H64](./slide_09_dinamica_e_custo/1_custo_computacional_H16_H48_H64.png)

### 2.2 Fronteira MSE, Funil `Treino → Val → Teste` e Custo do Campeão (**27 épocas em 21,5s**)
> [!NOTE]
> **Por que no painel central ("Funil de Generalização") a Acurácia de Treino (`82,8%`) é ligeiramente menor que a de Validação (`87,1%`) e Teste (`85,4%`)?**
> 1. A acurácia de **Treino** é calculada *online durante a época* (enquanto os pesos ainda estão sendo alterados exemplo a exemplo), enquanto **Validação e Teste** são medidos ao final da época com os pesos já estabilizados (`best_weights`).
> 2. Além disso, a Parada Antecipada interrompe o treino na época 22 **antes** de a rede decorar o conjunto de treino (`99%+`), preservando fronteiras suaves que generalizam melhor!

![Custo e Funil Campeão](./slide_09_dinamica_e_custo/2_custo_e_funil_campeao_85.4pct.png)

---

## 3. `slide_10_e_11_parada_antecipada_curvas/` (Para os Slides 10 e 11)

Curvas de evolução do **MSE e Acurácia por época** (linha verde = `Melhor Época` de menor MSE de validação; estrela vermelha `★` = resultado nos 130 dados fixos de teste):

| Arquivo | Cenário | Resultado no Teste | Como usar nos Slides 10 e 11 |
| :--- | :--- | :--- | :--- |
| `1_curva_campeao_momentum_com_es_85.4pct.png` | **Campeão Oficial (`M=0.85, P=5`)** | **85,38%** (`21,6s`, 27 ép.) | Curva principal de parada antecipada ideal (parou na época 27, melhor época 22). |
| `2_curva_gd_puro_com_es_83.8pct.png` | **GD Puro (`M=0.00, P=10`)** | **83,85%** (`95,0s`, 119 ép.) | Comparativo sem momentum: mostra a descida lenta que exige **4,4x mais épocas e tempo**. |
| `3_curva_sem_estratificacao_80.8pct.png` | **Sem Estratificação (`STRATIFIED=false`)** | **80,77%** (**-4,61 pp**) | **Ambiente Dificultado 1 (Slide 10):** desbalanceamento de classes na validação prejudica o corte. |
| `4_curva_com_ruido_15pct_76.9pct.png` | **Com Ruído 15% (`NOISE_RATE=0.15`)** | **76,92%** (**-8,46 pp**) | **Ambiente Dificultado 2 (Slide 10):** inversão de 15% dos pixels em treino/validação desloca o limiar de generalização. |
| `5_curva_sem_parada_antecipada_80ep_80.0pct.png` | **Sem Parada Antecipada (`80 épocas`)** | **80,00%** (**-5,38 pp**) | **Contraste Obrigatório (Slide 11):** treino vai a `99,16%`, mas teste cai de `85,38%` para `80,00%` por *overfitting*. |

### 3.1 Campeão Oficial (`M=0.85, P=5, Estratificado`) — **85,38% no Teste** (27 épocas / `21,6s`)
> [!NOTE]
> **Por que o gráfico vai até a época 27 se a linha verde (`Melhor Época`) está na 22?**
> Porque `PATIENCE=5`: o algoritmo precisa observar **5 épocas seguidas sem melhora** no MSE de validação (épocas 23 a 27) para confirmar que atingiu o mínimo e parar. Ao encerrar na época 27, o nosso orquestrador faz **rollback automático** dos pesos para o snapshot salvo na **época 22** (onde a estrela `★` de teste é avaliada).

![Curva Campeão](./slide_10_e_11_parada_antecipada_curvas/1_curva_campeao_momentum_com_es_85.4pct.png)

### 3.2 Gradiente Descendente Puro (`M=0.00, P=10, Estratificado`) — **83,85% no Teste** (119 épocas / `95,0s` — 4,4x mais lento)
![Curva GD Puro](./slide_10_e_11_parada_antecipada_curvas/2_curva_gd_puro_com_es_83.8pct.png)

### 3.3 Ambiente Dificultado 1: Sem Estratificação (`STRATIFIED=false`) — **80,77% no Teste (**-4,61 pp**)**
![Curva Sem Estratificação](./slide_10_e_11_parada_antecipada_curvas/3_curva_sem_estratificacao_80.8pct.png)

### 3.4 Ambiente Dificultado 2: Com Ruído Bipolar de 15% (`NOISE_RATE=0.15`) — **76,92% no Teste (**-8,46 pp**)**
> [!NOTE]
> **Por que aqui o MSE de Treino/Validação (~0,08) é maior do que o MSE de Teste (`0,0575`)?**
> Porque invertemos `15%` dos pixels (**-1 ↔ +1**) nos dados de **Treino e Validação** para dificultar o aprendizado e estressar a parada antecipada, enquanto os **130 exemplos de Teste** são o conjunto oficial limpo da professora.

![Curva Com Ruído 15%](./slide_10_e_11_parada_antecipada_curvas/4_curva_com_ruido_15pct_76.9pct.png)

### 3.5 Sem Parada Antecipada (`VAL_RATIO=0.0`, 80 épocas fixas) — **80,00% no Teste (**-5,38 pp** por Overfitting)**
> [!IMPORTANT]
> **Por que este gráfico (3.5) NÃO tem a linha laranja de Validação nem a linha verde de `Melhor Época`?**
> Porque este é justamente o experimento **Sem Parada Antecipada (`VAL_RATIO=0.0`)**: sem parada antecipada não existe conjunto de validação separado (todos os `1.196` exemplos disponíveis vão direto para o treino por 80 épocas fixas).
> - Repare que a curva azul de treino cola em **0,0061 de MSE (`99,16%` de acurácia — decorou o treino)**, mas quando avaliamos os pesos finais da época 80 nos 130 dados de teste (estrela vermelha `★`), a acurácia **despenca para `80,00%` (`MSE 0,0617`)** — uma perda de **-5,38 pp** em relação ao modelo com Parada Antecipada (`85,38%`)!

![Curva Sem Parada Antecipada](./slide_10_e_11_parada_antecipada_curvas/5_curva_sem_parada_antecipada_80ep_80.0pct.png)

---

## 4. `slide_12_matrizes_de_confusao/` (Para o Slide 12)

Matrizes de confusão nos **130 exemplos fixos de teste** e relatórios com o top confusões:

> [!NOTE]
> **Por que o valor máximo na diagonal principal é `5`?**
> Porque o conjunto de teste fixo possui **130 exemplos = 5 alfabetos completos de 26 letras** (`26 × 5 = 130`). Portanto, cada letra aparece **exatamente 5 vezes** no teste (`5` na diagonal = `100%` de acerto naquela letra).

| Arquivo | O que mostra / Como usar no Slide 12 |
| :--- | :--- |
| `1_matriz_campeao_com_es_85.4pct.png` | **Matriz Principal (Com Parada Antecipada):** `111/130` acertos (**85,38%**). 100% em `Z, Y, V, T, S`; erros restritos a letras parecidas (`A/B → R`, `H → M`, `I → J`). |
| `2_matriz_sem_parada_antecipada_80.0pct.png` | **Matriz Comparativa (Sem Parada Antecipada - 80 ép.):** `104/130` acertos (**80,00%**). Mostra os 7 erros extras gerados pela saturação dos pesos. |
| `3_matriz_gd_puro_com_es_83.8pct.png` | **Matriz GD Puro (`M=0.00`):** `109/130` acertos (**83,85%**). |
| `4_matriz_sem_estratificacao_80.8pct.png` / `5_matriz_com_ruido_15pct_76.9pct.png` | Matrizes opcionais dos cenários dificultados (`105/130` e `100/130`). |
| `1_relatorio_...txt` a `3_relatorio_...txt` | Resumo em texto com acurácia por letra e top 5 confusões de cada matriz para citar na fala. |

### 4.1 Campeão Com Parada Antecipada (`85,38%` — 111/130 acertos)
![Matriz Campeão](./slide_12_matrizes_de_confusao/1_matriz_campeao_com_es_85.4pct.png)

### 4.2 Sem Parada Antecipada — 80 épocas (`80,00%` — 104/130 acertos, **-5,38 pp**)
![Matriz Sem Parada Antecipada](./slide_12_matrizes_de_confusao/2_matriz_sem_parada_antecipada_80.0pct.png)

### 4.3 Gradiente Descendente Puro Sem Momentum (`83,85%` — 109/130 acertos)
![Matriz GD Puro](./slide_12_matrizes_de_confusao/3_matriz_gd_puro_com_es_83.8pct.png)

### 4.4 Sem Estratificação (`80,77%`) e Com Ruído 15% (`76,92%`)
![Matriz Sem Estratificação](./slide_12_matrizes_de_confusao/4_matriz_sem_estratificacao_80.8pct.png)

![Matriz Com Ruído](./slide_12_matrizes_de_confusao/5_matriz_com_ruido_15pct_76.9pct.png)

---

## 5. `saidas_oficiais_edisciplinas/` (Arquivos Obrigatórios da Entrega)

Uma subpasta para cada um dos 5 modelos finais acima, já com o `.json` e os **5 arquivos exigidos na especificação**:
1. `hiperparametros.txt` (arquitetura, inicialização, divisão treino/val/teste e métricas)
2. `pesos_iniciais.csv` ((`_V.csv`, `_W.csv`))
3. `pesos_finais.csv` ((`_V.csv`, `_W.csv`))
4. `historico_erro_treinamento.csv` (MSE por época)
5. `saidas_teste_continuas.csv` e `saidas_teste_discretas.csv` (saídas para os 130 exemplos de teste)
