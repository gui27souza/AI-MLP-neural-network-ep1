# Guia de Organização do Grupo — Entrega 1 (MLP)

> **Data limite oficial da entrega:** 25 de novembro até 23h55 (via e-Disciplinas)
> **Nota:** Esta pasta (`planejamento/`) serve como guia interno do grupo durante a preparação dos slides, gravação do vídeo e preenchimento do checklist (será removida antes do merge final para entrega).

---

## Índice dos Documentos de Apoio

1. **[`00_visao_geral_e_plano.md`](./00_visao_geral_e_plano.md)**
   Resumo das refatorações realizadas no código (separação dos 130 exemplos fixos de teste, Early Stopping guiado pelo MSE de validação e modularização dos métodos da MLP/Layer).
2. **[`01_mapeamento_slides_e_codigo.md`](./01_mapeamento_slides_e_codigo.md)**
   Tabela direta relacionando cada slide do `docs/template-slides.pptx` (**Slides 2 a 13**) com o arquivo e linhas exatas do código-fonte.
3. **[`03_trechos_codigo_slides.md`](./03_trechos_codigo_slides.md)**
   **Pronto para copiar pros slides:** Todos os recortes de código refatorados para colar nos **Slides 2, 3, 4, 5, 6, 7 e 10/11**, com dicas do que explicar no vídeo em cada slide.
4. **[`02_features_adicionais_e_checklist.md`](./02_features_adicionais_e_checklist.md)**
   Levantamento das implementações adicionais já prontas no projeto para o **Slide 13** (*Momentum*, *Hold-out*, *Data Augmentation Bipolar*, *Best-Weights Rollback*, *Classificação Binária*).
5. **[`04_guia_preenchimento_checklist.md`](./04_guia_preenchimento_checklist.md)**
   **Pronto para copiar pro Excel:** Mapeamento linha a linha da planilha `docs/Checklist.xlsx` com os nomes dos arquivos e intervalos exatos de linhas.
6. **[`05_roteiro_seus_slides_6_a_11.md`](./05_roteiro_seus_slides_6_a_11.md)**
   Roteiro detalhado dos **Slides 6 a 11** (*Atualização de pesos*, *Reconhecimento de caractere*, *Determinação de parâmetros*, *Dinâmica de treinamento/teste* e *Parada antecipada*) + comandos dos experimentos finais.

---

## Resumo Rápido por Slide (`docs/template-slides.pptx`)

| Slide | Tema do Slide | Onde está o código / material pronto |
| :---: | :--- | :--- |
| **Slide 1** | Capa (Nomes + Nº USP + Apresentadores) | Preencher no `template-slides.pptx` |
| **Slide 2** | Estruturas de dados e programação (Entrada=120, Saída=26, Oculta=$p$, Pesos $V$ e $W$) | Ver recortes em [`03_trechos_codigo_slides.md`](./03_trechos_codigo_slides.md#slide-2--estruturas-de-dados-e-programação) |
| **Slide 3** | Feedforward e cálculo do erro da rede (MSE) | Ver recortes em [`03_trechos_codigo_slides.md`](./03_trechos_codigo_slides.md#slide-3--feedforward-e-cálculo-do-erro-da-rede) |
| **Slide 4** | Backpropagation – cálculo do erro na camada de saída ($\delta_k$) | Ver recortes em [`03_trechos_codigo_slides.md`](./03_trechos_codigo_slides.md#slide-4--backpropagation-cálculo-do-erro-para-retropropagação-camada-de-saída) |
| **Slide 5** | Backpropagation – contribuição do erro na camada escondida ($\delta_j$) | Ver recortes em [`03_trechos_codigo_slides.md`](./03_trechos_codigo_slides.md#slide-5--backpropagation-contribuição-do-erro-na-camada-escondida) |
| **Slide 6** | Atualização de pesos sinápticos ($\Delta W, \Delta V$) | Ver recortes em [`03_trechos_codigo_slides.md`](./03_trechos_codigo_slides.md#slide-6-seu-slide--atualização-de-pesos-sinápticos) e [`05_roteiro_seus_slides_6_a_11.md`](./05_roteiro_seus_slides_6_a_11.md#slide-6--atualização-de-pesos-sinápticos) |
| **Slide 7** | Resposta de reconhecimento de caractere (*Winner-Takes-All* / `argmax`) | Ver recortes em [`03_trechos_codigo_slides.md`](./03_trechos_codigo_slides.md#slide-7-seu-slide--resposta-de-reconhecimento-de-caractere) e [`05_roteiro_seus_slides_6_a_11.md`](./05_roteiro_seus_slides_6_a_11.md#slide-7--resposta-de-reconhecimento-de-caractere) |
| **Slide 8** | Determinação de parâmetros (CLI + Busca em Grade) | Ver [`05_roteiro_seus_slides_6_a_11.md`](./05_roteiro_seus_slides_6_a_11.md#slide-8--determinação-de-parâmetros) + gráficos do Grid Search |
| **Slide 9** | Dinâmica de treinamento e teste (Hardware + Tabelas de Tempo $\times$ Épocas) | Ver [`05_roteiro_seus_slides_6_a_11.md`](./05_roteiro_seus_slides_6_a_11.md#slide-9--dinâmica-de-treinamento-e-teste) |
| **Slide 10** | Treinamento e parada antecipada (Execuções no terminal + Divisão 962 / 234 / 130) | Ver [`05_roteiro_seus_slides_6_a_11.md`](./05_roteiro_seus_slides_6_a_11.md#slide-10--treinamento-e-parada-antecipada) |
| **Slide 11** | Parada antecipada (Gráficos de MSE + Ambiente dificultado) | Ver [`05_roteiro_seus_slides_6_a_11.md`](./05_roteiro_seus_slides_6_a_11.md#slide-11--parada-antecipada-gráficos-e-ambiente-dificultado) |
| **Slide 12** | Matriz de confusão (Com vs. Sem Parada Antecipada e análise de confusões) | Usar `confusion_matrix.png` e `evaluation_report.txt` das rodadas oficiais |
| **Slide 13** | Implementações adicionais (Momentum, Hold-out, Data Augmentation Bipolar, Rollback) | Ver [`02_features_adicionais_e_checklist.md`](./02_features_adicionais_e_checklist.md) |
