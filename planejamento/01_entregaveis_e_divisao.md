# 1. O que Entregar + Regras Críticas + Divisão de Slides

> **Integrantes do Grupo (EACH-USP | IA 2026 — Entrega 1: MLP):**
> - **Guilherme Oliveira de Souza** — Nº USP: `14671388`
> - **Paulo Rogerio Andrade de Oliveira** — Nº USP: `9365783`
> - **Victor Hugo Antunes Valadares** — Nº USP: `14760317`


## Checklist Final de Postagem (e-Disciplinas — Prazo: 25/nov até 23h55)
- [ ] **Identificação obrigatória em TODOS os artefatos** (`Checklist.xlsx` linha 26): incluir **Nome Completo + Nº USP de todos os integrantes** nos cabeçalhos dos arquivos `.py`, no `README.md`, nos slides (capa e encerramento) e em qualquer PDF.
- [ ] **Código-fonte documentado** (sem subir arquivos extras não solicitados).
- [ ] **Planilha `docs/Checklist.xlsx`** preenchida (copiar de `02_planilha_checklist.md`).
- [ ] **Vídeo 1 — MLP (5 a 20 minutos):**
  - [ ] **Janela com câmera (webcam) do aluno apresentando** visível durante a explicação (exigência explícita do PDF).
  - [ ] **Voz em velocidade 1x normal** (é proibido acelerar a fala — risco de nota 0; só é permitido acelerar o trecho visual da execução de treino no terminal).
  - [ ] Se o vídeo for hospedado fora do e-Disciplinas (YouTube/Drive), postar no e-Disciplinas um **arquivo `.pdf` contendo o link público + a minutagem (início e fim de cada tópico/slide)**.
- [ ] **5 Arquivos de saída para cada teste realizado** (gerados em `saidas_oficiais/`):
  1. `hiperparametros.txt` (arquitetura + inicialização + divisão treino/val/teste)
  2. `pesos_iniciais.csv` ((`_V.csv`, `_W.csv`))
  3. `pesos_finais.csv` ((`_V.csv`, `_W.csv`))
  4. `historico_erro_treinamento.csv` (erro MSE por época)
  5. `saidas_teste_continuas.csv` e `saidas_teste_discretas.csv` (saídas para os 130 dados de teste)

---

## Mapa Exato dos 14 Tópicos do Template (`docs/template-slides.pptx` — manter a ordem!)

> [!TIP]
> **Os 14 slides do template são um índice de tópicos na ordem obrigatória:** como a própria professora orienta no template (*"Você pode usar mais slides, entre os tópicos que coloquei ou depois para questões adicionais, mas NÃO altere a ordem de apresentação dos tópicos"*), desdobrem à vontade cada tópico em sub-slides (`2.1, 2.2`, `3.1, 3.2`, `8.1, 8.2`, `11.1, 11.2`, `12.1, 12.2`) mantendo o título do tópico no topo para que nenhum slide fique poluído!

| Tópico | Título Oficial no Template | Apresentador | O que apresentar | Referência no Repo |
| :---: | :--- | :---: | :--- | :--- |
| **1** | **Capa (MLP Implementações)** | **Paulo** | Nomes completos, Nº USP e indicação de quem apresenta no vídeo | `03_guia_consolidado_slides.md` |
| **2** | **Estruturas de dados e programação** | **Paulo** | Entrada (120 pixels), saída (26 classes), camada oculta ($p$) e matrizes de pesos ($V, W$) | `03_guia_consolidado_slides.md` |
| **3** | **Feedforward e cálculo do erro da rede** | **Paulo** | Código do `compute_net_input`, `forward`, `predict` e `compute_mse` | `03_guia_consolidado_slides.md` |
| **4** | **Backpropagation – cálculo do erro para retropropagação** | **Paulo** | Código do erro na camada de saída (`compute_output_deltas`, $\delta_k$) + derivada | `03_guia_consolidado_slides.md` |
| **5** | **Backpropagation – contribuição do erro na camada escondida** | **Paulo** | Código da contribuição na camada oculta (`compute_hidden_deltas`, $\delta_j$) | `03_guia_consolidado_slides.md` |
| **6** | **Atualização de pesos sinápticos** | **Guilherme** | Código do `update_weights` ($\Delta W, \Delta V$) e `_update_weights` | `03_guia...` / `04_roteiro...` |
| **7** | **Resposta de reconhecimento de caractere** | **Guilherme** | Código do `predict_class` (`argmax`) e `recognize_character` | `03_guia...` / `04_roteiro...` |
| **8** | **Determinação de parâmetros** | **Guilherme** | Como parametriza via CLI/scripts + quais parâmetros variam + gráficos/tabelas da busca | `03_guia...` / `04_roteiro...` |
| **9** | **Dinâmica de treinamento e teste** | **Guilherme** | Hardware + tabelas de tempo ($H \times \text{tempo} \times \text{épocas}$, $\eta \times \text{tempo} \times \text{épocas}$) | `03_guia...` / `04_roteiro...` |
| **10** | **Treinamento e parada antecipada** | **Guilherme** | **Execução no terminal/vídeo (mín. 2 execuções)** + erro treino/val/teste + divisão (`832/364/130`) | `03_guia...` / `04_roteiro...` |
| **11** | **Parada antecipada** | **Guilherme** | Gráficos das curvas de MSE + ambiente dificultado (ruído e/ou amostragem sem estratificação) | `03_guia...` / `04_roteiro...` |
| **12** | **Matriz de confusão** | **Victor** | Matrizes de confusão variando condições (parâmetros, com/sem parada antecipada) + análise | `03_guia_consolidado_slides.md` |
| **13** | **Implementações adicionais** | **Victor** | Hold-out estratificado, Momentum, Data Augmentation Bipolar, Rollback `best_weights`, Portas Lógicas/Teste de Mesa | `03_guia_consolidado_slides.md` |
| **14** | **Encerramento (MLP Implementações)** | **Victor** | Slide final com nomes e Nº USP | `03_guia_consolidado_slides.md` |
