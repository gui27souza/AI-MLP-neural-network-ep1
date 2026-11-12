# Estudo de Data Augmentation (Ruído Bipolar)

Experimento em grade executado com a técnica de aumento determinístico de dados de treino via inversão estocástica de pixels bipolares (-1.0 <-> +1.0).

## Parâmetros do Teste:
- **Técnica:** Inversão bipolar com ruído (`NOISE_RATE = 0.05`, 1 cópia por amostra).
- **Amostras de Validação:** Sem aumento (avaliação estritamente em dados reais limpos).

## Principais Conclusões:
- **Melhor Modelo:** Atingiu **85.89%** de acurácia com $H=48$, $LR=0.025$, $Momentum=0.85$ e $Patience=6$.
- **Comparativo:** Comparado à baseline limpa (que alcança ~87.4%), a injeção de ruído atuou como forte regularizador, aumentando a variabilidade e exigindo maior momentum para convergência.

## Conteúdo:
- `grid_search/`: Execução completa com os 27 modelos treinados, `summary.csv`, relatório Markdown e gráficos (`grid_search_insights.png`).
