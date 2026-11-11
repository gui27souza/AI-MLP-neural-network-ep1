# ===


class DatasetLoader:
    """
    Carregador e pré-processador do dataset oficial do EP1: CARACTERES COMPLETO.

    Estrutura do Dataset:
    - Entradas (X.txt): 1326 amostras x 120 pixels normalizados (matriz de imagem 10x12).
      O arquivo bruto contém uma vírgula residual ao final de cada linha gerando
      uma 121ª coluna espúria, que é sanitizada e descartada no carregamento.
    - Saídas (Y_letra.txt): 1326 rótulos categóricos de letras ('A' a 'Z'),
      distribuídos em 51 variações de fontes tipográficas para cada uma das 26 letras.

    Codificação Bipolar One-Hot:
    - Como a rede neural utiliza a função Sigmoide Bipolar com imagem contínua em (-1, +1),
      os alvos precisam residir no mesmo espaço de ativação.
    - Cada caractere é codificado em um vetor de 26 posições preenchido com -1.0
      (indicando inatividade/não-pertencimento à classe) e +1.0 no índice correspondente
      à letra (classe ativa).
    """

    @staticmethod
    def load_caracteres_completo(
        filepath_inputs: str = "data/CARACTERES COMPLETO/X.txt",
        filepath_labels: str = "data/CARACTERES COMPLETO/Y_letra.txt",
        val_ratio: float = 0.1,
    ) -> tuple[
        list[list[float]],  # train_inputs
        list[list[float]],  # train_targets
        list[list[float]],  # test_inputs
        list[list[float]],  # test_targets
        list[str],          # class_labels ('A'..'Z')
    ]:
        """
        Carrega, sanitiza, codifica e particiona o dataset de caracteres.

        Parâmetros:
        - filepath_inputs: Caminho do arquivo com os pixels dos caracteres.
        - filepath_labels: Caminho do arquivo com os caracteres ('A' a 'Z').
        - val_ratio: Proporção de amostras reservadas para o conjunto de teste/validação (ex: 0.1 = 10%).

        Retorna:
        - (train_inputs, train_targets, test_inputs, test_targets, class_labels)
        """
        # 1. Leitura e sanitização dos pixels de entrada
        # Descarte da 121ª coluna residual gerada por vírgula no fim da linha
        with open(filepath_inputs, "r", encoding="utf-8-sig") as f:
            all_inputs = [
                [float(pixel) for pixel in line.strip().split(",")[:120]]
                for line in f
                if line.strip()
            ]

        # 2. Leitura dos rótulos de classe ('A', 'B', ..., 'Z')
        with open(filepath_labels, "r", encoding="utf-8-sig") as f:
            all_labels = [line.strip() for line in f if line.strip()]

        # 3. Mapeamento das 26 classes do alfabeto
        class_labels = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
        label_to_index = {label: index for index, label in enumerate(class_labels)}

        # 4. Codificação One-Hot Bipolar (-1.0 e +1.0)
        all_targets: list[list[float]] = []
        for label in all_labels:
            target_vector = [-1.0] * len(class_labels)
            if label in label_to_index:
                target_vector[label_to_index[label]] = 1.0
            all_targets.append(target_vector)

        # 5. Particionamento formal entre Treino e Teste/Validação
        num_test_samples = int(len(all_inputs) * val_ratio)
        num_train_samples = len(all_inputs) - num_test_samples

        train_inputs = all_inputs[:num_train_samples]
        train_targets = all_targets[:num_train_samples]

        test_inputs = all_inputs[num_train_samples:]
        test_targets = all_targets[num_train_samples:]

        return train_inputs, train_targets, test_inputs, test_targets, class_labels
