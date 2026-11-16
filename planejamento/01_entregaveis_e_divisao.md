# 1. O que Entregar + Regras Críticas + Divisão de Slides

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
  2. `pesos_iniciais.csv` (`+ _V.csv, _W.csv`)
  3. `pesos_finais.csv` (`+ _V.csv, _W.csv`)
  4. `historico_erro_treinamento.csv` (erro MSE por época)
  5. `saidas_teste_continuas.csv` e `saidas_teste_discretas.csv` (saídas para os 130 dados de teste)

---

## Mapa Exato dos 14 Slides (`docs/template-slides.pptx` — não alterar a ordem!)

| Slide | Título Oficial no Template | O que apresentar | Referência no Repo |
| :---: | :--- | :--- | :--- |
| **1** | **Capa (MLP Implementações)** | Nomes completos, Nº USP e indicação de quem apresenta no vídeo | — |
| **2** | **Estruturas de dados e programação** | Entrada (120 pixels), saída (26 classes), camada oculta ($p$) e matrizes de pesos ($V, W$) | `03_codigo_slides_2_a_7.md` |
| **3** | **Feedforward e cálculo do erro da rede** | Código do `compute_net_input`, `forward`, `predict` e `compute_mse` | `03_codigo_slides_2_a_7.md` |
| **4** | **Backpropagation – cálculo do erro para retropropagação** | Código do erro na camada de saída (`compute_output_deltas`, $\delta_k$) + derivada | `03_codigo_slides_2_a_7.md` |
| **5** | **Backpropagation – contribuição do erro na camada escondida** | Código da contribuição na camada oculta (`compute_hidden_deltas`, $\delta_j$) | `03_codigo_slides_2_a_7.md` |
| **6** | **Atualização de pesos sinápticos** | Código do `update_weights` ($\Delta W, \Delta V$) e `_update_weights` | `03_codigo...` + `04_conteudo...` |
| **7** | **Resposta de reconhecimento de caractere** | Código do `predict_class` (`argmax`) e `recognize_character` | `03_codigo...` + `04_conteudo...` |
| **8** | **Determinação de parâmetros** | Como parametriza via CLI/scripts + quais parâmetros variam + gráficos/tabelas da busca | `04_conteudo_slides_6_a_12.md` |
| **9** | **Dinâmica de treinamento e teste** | Hardware + tabelas de tempo ($H \times \text{tempo} \times \text{épocas}$, $\eta \times \text{tempo} \times \text{épocas}$) | `04_conteudo_slides_6_a_12.md` |
| **10** | **Treinamento e parada antecipada** | **Execução no terminal/vídeo (mín. 2 execuções)** + erro treino/val/teste + divisão dos conjuntos (`962/234/130`) | `04_conteudo_slides_6_a_12.md` |
| **11** | **Parada antecipada** | Gráficos das curvas de MSE + ambiente dificultado (ruído e/ou amostragem sem estratificação) | `04_conteudo_slides_6_a_12.md` |
| **12** | **Matriz de confusão** | Matrizes de confusão variando condições (parâmetros, com/sem parada antecipada) + análise | `04_conteudo_slides_6_a_12.md` |
| **13** | **Implementações adicionais** | Hold-out estratificado, Momentum, Data Augmentation Bipolar, Rollback `best_weights`, Portas Lógicas/Teste de Mesa | `04_conteudo_slides_6_a_12.md` |
| **14** | **Encerramento (MLP Implementações)** | Slide final com nomes e Nº USP | — |
