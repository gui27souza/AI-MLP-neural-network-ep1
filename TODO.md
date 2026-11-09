## Trazer da versão do colab

3. Exportação dos 5 Arquivos Oficiais de Entrega
    - O que faz: Gera a pasta saidas_oficiais/ com os arquivos que a professora exige no PDF:
        - hiperparametros.txt
        - pesos_iniciais.csv e pesos_finais.csv (formato matricial de Fausett)
        - historico_erro_treinamento.csv
        - saidas_teste_continuas.csv e saidas_teste_discretas.csv
4. Aumento de Dados com Ruído (Data Augmentation Bipolar)
    - O que faz: Cria cópias ruidosas no treino invertendo aleatoriamente pixels (-1 ↔ +1) para a rede ficar mais robusta.
5. Visualização Gráfica e Matriz de Confusão (para o vídeo/slides)
    - O que faz: Gráficos de curvas de aprendizado (MSE e Acurácia) e a matriz de confusão 26 × 26 das letras.
6. Otimizador de Hiperparâmetros (Grid Search Adaptativo)
    - O que faz: Script que testa combinações de taxa de aprendizado, quantidade de neurônios e momentum para encontrar os parâmetros ideais.

