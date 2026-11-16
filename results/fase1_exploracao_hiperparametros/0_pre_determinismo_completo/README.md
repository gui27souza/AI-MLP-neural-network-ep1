# Experimentos Pré-Determinismo Completo

Modelos e buscas em grade executados na fase exploratória inicial da MLP, antes do fechamento do determinismo estrito (seed de pesos isolada, shuffling determinístico e splits fixos).

Possuem alto valor de estudo e análise para o relatório final:
- Evidenciaram o comportamento das topologias (ex: consolidação de $H=48$ como ponto de equilíbrio frente a $H=16$ e $H=64$).
- Balizaram as primeiras faixas ótimas de taxa de aprendizado ($\eta \in [0.008, 0.025]$) e momentum ($\alpha \in [0.15, 0.85]$).

## Conteúdo:
- `grid_searches/`: As 4 rodadas completas de exploração em grade anteriores.
- `avaliacoes_modelos/`: Gráficos, métricas e matrizes de confusão dos destaques dessa fase (`gs_acc77.8_h16`, `gs_acc84.9_h64`, `gs_acc86.4_h48`, `gs_acc86.6_h48`).
