import random


class DatasetLoader:
    """
    Carregador e pré-processador do dataset oficial do EP1: CARACTERES COMPLETO.

    Estrutura do Dataset (data/CARACTERES COMPLETO/):
    - Total: 1.326 amostras organizadas em 51 blocos sequenciais do alfabeto ('A' a 'Z'),
      onde cada bloco de 26 amostras corresponde a uma variação tipográfica completa.
    - Entradas (X.txt): 1.326 linhas x 120 pixels bipolares (-1 e +1, matriz 10x12).
      O arquivo bruto contém uma vírgula residual ao final de cada linha gerando
      uma 121ª coluna vazia, que é sanitizada no carregamento.
    - Saídas (Y_letra.txt): 1.326 rótulos categóricos ('A' a 'Z'), codificados em
      vetores One-Hot Bipolar de 26 posições (+1.0 na classe ativa e -1.0 nas demais).

    Particionamento Oficial (conforme data/CARACTERES COMPLETO/README.md):
    1. Conjunto de Teste (Fixo): Os últimos 130 exemplos (índices 1196 a 1325, equivalentes
       aos últimos 5 alfabetos completos: 5 x 26 = 130) pertencem exclusivamente ao Teste.
    2. Base de Desenvolvimento (Primeiros 1.196 exemplos = 46 alfabetos completos):
       Dividida entre Conjunto de Treinamento e Conjunto de Validação segundo `val_ratio`.
       - Por padrão (`stratified=True`), o corte é alinhado em múltiplos de 26 amostras
         (alfabetos inteiros), garantindo estratificação exata entre todas as letras A-Z.
       - Opcionalmente (`stratified=False`), realiza amostragem aleatória sem estratificação
         (útil para os experimentos de parada antecipada do Slide 11).
    """

    NUM_TEST_SAMPLES: int = 130
    ALPHABET_SIZE: int = 26

    @classmethod
    def load_caracteres_completo(
        cls,
        filepath_inputs: str = "data/CARACTERES COMPLETO/X.txt",
        filepath_labels: str = "data/CARACTERES COMPLETO/Y_letra.txt",
        val_ratio: float = 0.2,
        stratified: bool = True,
        seed: int = 42,
    ) -> tuple[
        list[list[float]],  # train_inputs
        list[list[float]],  # train_targets
        list[list[float]],  # val_inputs
        list[list[float]],  # val_targets
        list[list[float]],  # test_inputs (130 fixos)
        list[list[float]],  # test_targets (130 fixos)
        list[str],          # class_labels ('A'..'Z')
    ]:
        """
        Carrega, sanitiza, codifica e particiona o dataset em Treino, Validação e Teste.

        Retorna:
        - (train_inputs, train_targets, val_inputs, val_targets, test_inputs, test_targets, class_labels)
        """
        # 1. Leitura e sanitização dos pixels de entrada (120 colunas por amostra)
        with open(filepath_inputs, "r", encoding="utf-8-sig") as f:
            all_inputs = [
                [float(pixel) for pixel in line.strip().split(",")[:120]]
                for line in f
                if line.strip()
            ]

        # 2. Leitura dos rótulos de classe ('A', 'B', ..., 'Z')
        with open(filepath_labels, "r", encoding="utf-8-sig") as f:
            all_labels = [line.strip() for line in f if line.strip()]

        # 3. Mapeamento das 26 classes do alfabeto e codificação One-Hot Bipolar (-1.0 e +1.0)
        class_labels = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
        label_to_index = {label: index for index, label in enumerate(class_labels)}

        all_targets: list[list[float]] = []
        for label in all_labels:
            target_vector = [-1.0] * len(class_labels)
            if label in label_to_index:
                target_vector[label_to_index[label]] = 1.0
            all_targets.append(target_vector)

        # 4. Separação fixa do Conjunto de Teste: últimos 130 exemplos (5 alfabetos completos)
        dev_limit = len(all_inputs) - cls.NUM_TEST_SAMPLES
        dev_inputs = all_inputs[:dev_limit]
        dev_targets = all_targets[:dev_limit]

        test_inputs = all_inputs[dev_limit:]
        test_targets = all_targets[dev_limit:]

        # 5. Particionamento da Base de Desenvolvimento (1.196 amostras) em Treino e Validação
        if val_ratio <= 0.0:
            return (
                dev_inputs,
                dev_targets,
                [],
                [],
                test_inputs,
                test_targets,
                class_labels,
            )

        if stratified:
            # Alinha o número de amostras de validação em múltiplos de 26 (alfabetos completos)
            total_dev_alphabets = len(dev_inputs) // cls.ALPHABET_SIZE
            val_alphabets = max(1, round(total_dev_alphabets * val_ratio))
            num_val_samples = val_alphabets * cls.ALPHABET_SIZE
            num_train_samples = len(dev_inputs) - num_val_samples

            train_inputs = dev_inputs[:num_train_samples]
            train_targets = dev_targets[:num_train_samples]
            val_inputs = dev_inputs[num_train_samples:]
            val_targets = dev_targets[num_train_samples:]
        else:
            # Amostragem aleatória sem estratificação (Slide 11)
            rng = random.Random(seed)
            indices = list(range(len(dev_inputs)))
            rng.shuffle(indices)

            num_val_samples = max(1, int(round(len(dev_inputs) * val_ratio)))
            num_train_samples = len(dev_inputs) - num_val_samples

            train_idx = indices[:num_train_samples]
            val_idx = indices[num_train_samples:]

            train_inputs = [dev_inputs[i] for i in train_idx]
            train_targets = [dev_targets[i] for i in train_idx]
            val_inputs = [dev_inputs[i] for i in val_idx]
            val_targets = [dev_targets[i] for i in val_idx]

        return (
            train_inputs,
            train_targets,
            val_inputs,
            val_targets,
            test_inputs,
            test_targets,
            class_labels,
        )

# ==============================================================================
# Integrantes do Grupo (EACH-USP | Inteligência Artificial 2026):
# - Guilherme Oliveira de Souza (Nº USP: 14671388)
# - Paulo Rogerio Andrade de Oliveira (Nº USP: 9365783)
# - Victor Hugo Antunes Valadares (Nº USP: 14760317)
# ==============================================================================
