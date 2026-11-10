## TO-DO para entrega

- Exportação dos 5 Arquivos Oficiais de Entrega
    - O que faz: Gera a pasta saidas_oficiais/ com os arquivos que a professora exige no PDF:
        - hiperparametros.txt
        - pesos_iniciais.csv e pesos_finais.csv (formato matricial de Fausett)
        - historico_erro_treinamento.csv
        - saidas_teste_continuas.csv e saidas_teste_discretas.csv

- Aumento de Dados com Ruído (Data Augmentation Bipolar)
    - O que faz: Cria cópias ruidosas no treino invertendo aleatoriamente pixels (-1 ↔ +1) para a rede ficar mais robusta.

- Visualização Gráfica e Matriz de Confusão (para o vídeo/slides)
    - O que faz: Gráficos de curvas de aprendizado (MSE e Acurácia) e a matriz de confusão 26 × 26 das letras.

## Refinamentos

- Critério de Seleção do Modelo (Acurácia x MSE):
    - Mudar o salvamento do `best_model` e o acionamento do Early Stopping para focar na maior Acurácia de Validação.
    - Usar o menor MSE de Validação como critério de desempate caso haja empate na acurácia.

- Script de Grid Search (Busca Automática):
    - Criar o script utilitário que itera sobre combinações de hiper-parâmetros
    - Executar o treino para cada par e salvar no JSON apenas a melhor combinação final

- Ajuste da Proporção Treino x Validação:
    - Testar variações além do atual 90/10 (ex: 80/20 ou 75/25) para ter um conjunto de validação estatisticamente mais representativo que as 130 amostras atuais.

- Decaimento da Taxa de Aprendizado (Learning Rate Decay):
    - Implementar a redução do eta ao longo das épocas ( eta(t) = eta_0/(1 + k * t) ) ou a redução em platôs (reduzir eta pela metade caso a validação estagne por 2 épocas).


## Sugestões Complementares

- Inicialização Nguyen-Widrow
    - Substituir o intervalo aleatório simples [-0.2, 0.2] pela fórmula da Faussett (beta = 0.7 * p^{1/n}), que ajusta a escala da norma dos pesos da camada oculta.

- Truque contra 'Flat Spots'
    - Adicionar uma pequena constante (+0.1) na derivada da sigmóide bipolar durante o cálculo do erro para impedir que o gradiente mude para zero e trave o aprendizado quando a saída se aproxima de +/- 1.

- Regularização L2 (Weight Decay)
    - Incluir um termo de penalidade (-eta * lambda * w_ij) na regra de atualização para evitar que os pesos cresçam descontroladamente e reduza o *overfitting*.
