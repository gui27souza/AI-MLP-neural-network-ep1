# 4. Roteiro de Gravação — Guilherme (Sincronizado Slide a Slide com `Slides - MLP (2).pdf`)

> **Integrantes do Grupo (EACH-USP | IA 2026 — Entrega 1: MLP):**
> - **Guilherme Oliveira de Souza** — Nº USP: `14671388`
> - **Paulo Rogerio Andrade de Oliveira** — Nº USP: `9365783`
> - **Victor Hugo Antunes Valadares** — Nº USP: `14760317`

Este roteiro já está **100% sincronizado página por página (Páginas 6 a 22 do seu PDF)** para você só ir apertando a **seta para a direita (`➔`)** nos marcadores **[👉 PASSA SLIDE]** enquanto grava, sem precisar dar `Alt+Tab` para o VS Code (já que todos os trechos de código comentados estão nos próprios slides!).

---

## 🎬 Preparação Rápida para a Página 18 (Clipe Acelerado das 2 Execuções no Terminal)

Para colocar no lugar do `TO-DO` da **Página 18**, basta gravar a janela do terminal rodando os 2 comandos abaixo e acelerar o vídeo na edição (~10s para cada execução, mantendo sua voz em `1x`):

```bash
# Execução 1 — Campeão com Momentum 0.85 (27 épocas em ~21s -> acelerado 2x vira ~10s):
make train NUM_EPOCHS=120 HIDDEN_NEURONS=48 LEARN_RATE=0.005 MOMENTUM=0.85 PATIENCE=5 VAL_RATIO=0.30 STRATIFIED=true

# Execução 2 — GD Puro sem Momentum 0.00 (119 épocas em ~95s -> acelerado 8x vira ~12s):
make train NUM_EPOCHS=150 HIDDEN_NEURONS=48 LEARN_RATE=0.005 MOMENTUM=0.00 PATIENCE=10 VAL_RATIO=0.30 STRATIFIED=true
```

---

## 🎙️ Roteiro Falado Sincronizado (Páginas 6 a 22 — ~6 a 6,5 min em 1x)

---

### ⏱️ **[0:00 – 1:05] Tópico 6: Atualização de Pesos Sinápticos (Páginas 6 e 7)**

**[PÁGINA 6 — Atualização de pesos sinápticos: `MLP._update_weights`]**

*"Pessoal, dando sequência na explicação do nosso código, agora entramos na **atualização dos pesos sinápticos**.*
- *Um cuidado fundamental que tomamos na implementação — seguindo rigorosamente os Passos 6, 7 e 8 do algoritmo de Fausett — está na ordem modular dentro do método `_update_weights` da classe `MLP`:*
- *Primeiro calculamos os deltas da camada de saída ($\delta_k$), depois retropropagamos a contribuição do erro para a camada escondida ($\delta_j$) usando a matriz de pesos $W$ **ainda original**, e somente na terceira linha aplicamos a alteração dos pesos e biases em todas as camadas.*

**[👉 PASSA SLIDE $\rightarrow$ PÁGINA 7 — Atualização de pesos sinápticos: `Layer.update_weights`]**

*"Aqui no método `update_weights` da classe `Layer`, vemos como ocorre efetivamente o ajuste sináptico:*
- *Para cada neurônio `i_neuron`, atualizamos primeiro o seu **bias** somando a taxa de aprendizado `learn_rate` vezes o `delta_i` daquele neurônio, mais o termo de `momentum` multiplicado pela variação da iteração anterior (`last_delta_bias`).*
- *Em seguida, no laço interno, percorremos cada entrada `j_input` conectada ao neurônio — que na camada escondida são os 120 pixels $x_i$ e na camada de saída são as ativações ocultas $z_j$ salvas no feedforward — aplicando a Regra Delta Generalizada: $\Delta w = \alpha \cdot \delta \cdot x + \mu \cdot \Delta w(t-1)$.*
- *Quando `momentum` é `0.0`, executamos o Gradiente Descendente puro exigido na especificação; quando ativamos o `momentum`, adicionamos inércia para acelerar a convergência."*

---

### ⏱️ **[1:05 – 1:50] Tópico 7: Resposta de Reconhecimento de Caractere (Páginas 8 e 9)**

**[👉 PASSA SLIDE $\rightarrow$ PÁGINA 8 — Resposta de reconhecimento de caractere: `predict_class`]**

*"Passando para o procedimento de **resposta de reconhecimento de caractere**:*
- *Como a nossa camada de saída possui **26 neurônios** ativados pela função Sigmoide Bipolar, o método `predict` retorna um vetor contínuo de 26 valores reais no intervalo entre `-1` e `+1`.*
- *Para decidir qual letra foi reconhecida, implementamos a regra **Winner-Takes-All** no método `predict_class` usando `np.argmax(outputs)`: selecionamos o índice $k^*$ do neurônio de saída que apresentou a **maior ativação contínua**, garantindo uma decisão única e sem empates.*

**[👉 PASSA SLIDE $\rightarrow$ PÁGINA 9 — Resposta de reconhecimento de caractere: `recognize_character`]**

*"Na sequência, o método `recognize_character` recebe esse índice vencedor de `0 a 25` e o converte diretamente na letra correspondente de `'A'` a `'Z'`.*
- *Além disso, na persistência oficial dos testes, gravamos tanto o arquivo de **saídas contínuas** com os valores reais $y_k$ quanto o de **saídas discretas**, onde o neurônio vencedor recebe `+1.0` e os demais recebem `-1.0`."*

---

### ⏱️ **[1:50 – 3:20] Tópico 8: Determinação de Parâmetros (Páginas 10, 11, 12 e 13)**

**[👉 PASSA SLIDE $\rightarrow$ PÁGINA 10 — Determinação de parâmetros: CLI e Variáveis]**

*"Entrando na etapa experimental, explicamos agora a **determinação de parâmetros**.*
- *Toda a nossa arquitetura é parametrizada via linha de comando através do `Makefile` e de variáveis de ambiente lidas em `config.py`, sem precisar alterar o código-fonte.*
- *Como listado abaixo, durante a execução podemos variar livremente o número de neurônios ocultos (`HIDDEN_NEURONS`), a taxa de aprendizado (`LEARN_RATE`), o `MOMENTUM`, a proporção de validação (`VAL_RATIO`), o uso de amostragem estratificada (`STRATIFIED`), a paciência da parada antecipada (`PATIENCE`), o número de épocas e a taxa de ruído (`NOISE_RATE`).*

**[👉 PASSA SLIDE $\rightarrow$ PÁGINA 11 — Determinação de parâmetros: Busca em Funil]**

*"Para encontrar os melhores hiperparâmetros mantendo as últimas 130 amostras fixas para teste, estruturamos um **procedimento de busca em funil em 4 Grids progressivos**:*
- *No **Grid 1**, fizemos uma varredura ampla em 18 modelos que descartou `H=16` por underfitting (`66,8%` no teste) e apontou `H=48` como a melhor média (`81,9%` no teste).*
- *No **Grid 2**, avaliamos a proporção de validação entre 10%, 20% e 30%, onde `30%` (364 amostras = 14 alfabetos) estabilizou melhor o MSE de validação.*
- *No **Grid 3**, refinamos `H=48` e `64` com paciência 5, atingindo o nosso recorde de **85,38% no teste (111 acertos em 130)** com `H=48`, `LR=0.005` e `Momentum=0.85`.*
- *E no **Grid 4**, aplicamos um zoom fino ao redor do modelo campeão que manteve média de **84,19% no teste** e **87,33% na validação**, comprovando que encontramos um platô estável e não um pico isolado de sorte.*

**[👉 PASSA SLIDE $\rightarrow$ PÁGINA 12 — Determinação de parâmetros: Gráfico de Sensibilidade]**

*"Este gráfico de sensibilidade ilustra o impacto individual de cada parâmetro:*
- *No painel à esquerda, fica evidente por que o número de neurônios ocultos ($H$) é o parâmetro mais determinante: 16 neurônios não comportam a variabilidade tipográfica de 120 pixels em 26 classes, enquanto **$H=48$ (7.082 pesos)** salta mais de 15 pontos percentuais e supera inclusive $H=64$ (9.434 pesos).*

**[👉 PASSA SLIDE $\rightarrow$ PÁGINA 13 — Determinação de parâmetros: Heatmap LR x Momentum]**

*"E neste mapa de calor bivariado vemos a forte **interação entre Taxa de Aprendizado ($\eta$) e Momentum ($\mu$)**:*
- *Como nosso treinamento é online (832 ajustes de peso por época), combinar passo alto (`0.025`) com momentum alto (`0.90`) gera oscilação no MSE e derruba o teste para `78,5%`.*
- *Já o ponto ótimo no centro do mapa combina passo moderado (`0.005`) com inércia alta (`0.85`), cravando **85,4% no teste** e **87,1% na validação**."*

---

### ⏱️ **[3:20 – 4:30] Tópico 9: Dinâmica de Treinamento e Teste (Páginas 14, 15 e 16)**

**[👉 PASSA SLIDE $\rightarrow$ PÁGINA 14 — Dinâmica de treinamento e teste: Hardware]**

*"Passando para a **dinâmica de treinamento e teste**, todos os nossos experimentos no conjunto `CARACTERES COMPLETO` (com 832 amostras de treino, 364 de validação e 130 de teste fixo) foram executados em um desktop **Intel Core i5-14600KF a 5.30 GHz com 32 GB de RAM DDR5**.*

**[👉 PASSA SLIDE $\rightarrow$ PÁGINA 15 — Dinâmica: Tabela de Neurônios Ocultos + Gráfico]**

*"Analisando a relação entre a **quantidade de neurônios ocultos ($H$), épocas e tempo de execução**:*
- *Vemos na tabela e no gráfico de bolhas abaixo que o **custo por época cresce linearmente** com o tamanho da camada escondida: `0,31s` por época em $H=16$, `0,79s` em $H=48$ e `1,05s` em $H=64$.*
- *$H=48$ converge em menos épocas médias (`43,2 épocas`, `34,3 segundos`) e entrega a melhor acurácia tanto na validação (`87,16%` média) quanto no teste (`85,38%`), sendo o melhor custo-benefício computacional.*

**[👉 PASSA SLIDE $\rightarrow$ PÁGINA 16 — Dinâmica: Tabelas de Learn Rate e Momentum]**

*"Já mantendo $H=48$ fixo e variando a **Taxa de Aprendizado** e o **Momentum**:*
- *Na tabela superior, vemos que com taxa muito baixa (`0,001`) a rede leva em média **73,3 épocas (58,3 segundos)** para convergir, enquanto com `0,005` o tempo cai pela metade: **29 épocas (22,9 segundos)** com a maior média e pico no teste.*
- *Na tabela inferior, comparando diretamente o **Gradiente Descendente Puro (`M=0`)** com o nosso **Campeão com Momentum (`M=0,85`)**, o GD Puro precisou de **119 épocas e 95 segundos** para atingir `83,85%` no teste, enquanto o Momentum convergiu em apenas **27 épocas e 21,56 segundos** cravando **85,38%** — uma redução de **77% no tempo de treino (4,4 vezes mais rápido)**."*

---

### ⏱️ **[4:30 – 5:30] Tópico 10: Treinamento e Parada Antecipada (Páginas 17 e 18)**

**[👉 PASSA SLIDE $\rightarrow$ PÁGINA 17 — Treinamento e parada antecipada: Divisão dos Conjuntos]**

*"Entrando em **treinamento e parada antecipada**, detalhamos aqui como particionamos as `1.326` amostras (51 alfabetos) do `CARACTERES COMPLETO`:*
- *As **últimas 130 amostras (os 5 últimos alfabetos completos, índices 1196 a 1325)** foram isoladas de forma fixa exclusivamente para o **Conjunto de Teste**.*
- *Das `1.196` amostras restantes, separamos de forma estratificada **70% para o Conjunto de Treinamento (832 amostras = 32 alfabetos completos)** e **30% para o Conjunto de Validação (364 amostras = 14 alfabetos completos)**, monitorando o MSE de validação a cada época com rollback automático para os melhores pesos (`best_weights`).*

**[👉 PASSA SLIDE $\rightarrow$ PÁGINA 18 — Vídeo Acelerado do Terminal + Resumo das 2 Execuções]**

*"Acompanhando aqui no vídeo acelerado do nosso terminal as duas execuções comparativas com `H=48` e `LR=0.005`:*
- *Na **Execução 1 (Campeão com Momentum 0.85 e Paciência 5)**, o MSE cai de `0.0725` no treino e `0.0568` na validação na Época 1 até atingir o **menor MSE de validação na Época 22 (`0.03451`, com 87,09% de acurácia)**. Após 5 épocas sem melhora, o treino para na Época 27 em `21,56s`, restaura os pesos da Época 22 e atinge **MSE de teste `0.05241` e acurácia de 85,38% (111/130)**.*
- *Já na **Execução 2 (Gradiente Descendente Puro sem Momentum, Paciência 10)**, a descida do MSE é bem mais lenta: o mínimo de validação ocorre apenas na **Época 109 (`0.03513`)**, parando na Época 119 em `95 segundos` com **83,85% no teste (109/130)**."*

---

### ⏱️ **[5:30 – 6:30] Tópico 11: Parada Antecipada e Ambiente Dificultado (Páginas 19, 20, 21 e 22)**

**[👉 PASSA SLIDE $\rightarrow$ PÁGINA 19 — Parada antecipada: Curva do Campeão 85,38%]**

*"Visualizando agora os gráficos de **parada antecipada** minimizando o Erro Quadrático Médio (MSE):*
- *Nesta curva do modelo campeão, vemos o MSE de treino (azul) e de validação (laranja) caindo juntos até a linha vertical verde na **Época 22**, onde o MSE de validação estaciona em `0.03451` e a parada antecipada protege a rede, garantindo **85,38% no teste** (estrela vermelha).*

**[👉 PASSA SLIDE $\rightarrow$ PÁGINA 20 — Parada antecipada: Sem Parada Antecipada 80 Épocas]**

*"Em contraste, neste gráfico treinamos a mesma rede **sem parada antecipada por 80 épocas fixas**:*
- *O treino chega a `99,16%` de acerto com MSE próximo de zero (`0.0061`), mas a saturação dos pesos gera **overfitting** e derruba a acurácia no teste de **85,38% para 80,00% (-5,38 pontos percentuais)**.*

**[👉 PASSA SLIDE $\rightarrow$ PÁGINA 21 — Ambiente Dificultado 1: Sem Estratificação]**

*"Por fim, atendendo à proposta do template de **dificultar o ambiente de experimentação** para avaliar a parada antecipada sob condições adversas, testamos dois cenários:*
- *Primeiro, com **amostragem aleatória sem estratificação (`STRATIFIED=false`)**: o desbalanceamento das letras na validação desloca o ponto de corte para a época 31 e reduz a acurácia no teste em **-4,61 pontos percentuais (`80,77%`)**.*

**[👉 PASSA SLIDE $\rightarrow$ PÁGINA 22 — Ambiente Dificultado 2: Com Ruído Bipolar de 15%]**

*"E segundo, injetando **15% de ruído bipolar** (invertendo estocasticamente 15% dos pixels em treino e validação):*
- *As curvas de MSE ficam visivelmente mais elevadas e ruidosas, mas a parada antecipada ainda consegue identificar o ponto de estabilização na época 30 e preservar **76,92% de acurácia (100/130)** nos dados de teste. Agora passo a palavra para o Victor apresentar as Matrizes de Confusão e as implementações adicionais."*
