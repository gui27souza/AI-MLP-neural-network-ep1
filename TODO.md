## MLP

- Adicionar teto de execução com threshold de MSE
    - Executar X épocas ou enquanto MSE for maior que Y


## Docs

- Melhorar documentação dos passos críticos do Backpropagation
    - Adicionar fórmulas para melhor seguimento do entendimento

## Trazer da versão do colab

1. Termo de Momentum (μ) no Backpropagation
    - O que faz: Adiciona inércia na atualização dos pesos (Δwₜ = α · grad + μ · Δwₜ₋₁).
    - Onde entra: No Layer e MLP.
    - Impacto: Faz a rede convergir mais rápido e sem oscilar tanto.

2. Embaralhamento Estocástico (Shuffle por época)
    - O que faz: Embaralha a ordem das 1196 amostras no início de cada época.
    - Onde entra: No loop do Orchestrator.
    - Impacto: Impede que a rede fique viciada na sequência alfabética e ajuda a escapar de mínimos locais.

3. Critérios de Parada e Early Stopping
    - O que faz: Parar por MSE alvo (target_error) e parar quando o erro de teste parar de cair (patience), restaurando os melhores pesos observados.
    - Onde entra: No Orchestrator e MLP.

4. Exportação dos 5 Arquivos Oficiais de Entrega
    - O que faz: Gera a pasta saidas_oficiais/ com os arquivos que a professora exige no PDF:
        - hiperparametros.txt
        - pesos_iniciais.csv e pesos_finais.csv (formato matricial de Fausett)
        - historico_erro_treinamento.csv
        - saidas_teste_continuas.csv e saidas_teste_discretas.csv

5. Aumento de Dados com Ruído (Data Augmentation Bipolar)
    - O que faz: Cria cópias ruidosas no treino invertendo aleatoriamente pixels (-1 ↔ +1) para a rede ficar mais robusta.

6. Visualização Gráfica e Matriz de Confusão (para o vídeo/slides)
    - O que faz: Gráficos de curvas de aprendizado (MSE e Acurácia) e a matriz de confusão 26 × 26 das letras.

7. Otimizador de Hiperparâmetros (Grid Search Adaptativo)
    - O que faz: Script que testa combinações de taxa de aprendizado, quantidade de neurônios e momentum para encontrar os parâmetros ideais.

