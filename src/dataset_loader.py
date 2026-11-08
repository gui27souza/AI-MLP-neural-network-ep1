import csv
import numpy as np


class DatasetLoader:

    @staticmethod
    def load_csv(filepath: str, num_target_columns: int = 1, delimiter: str = ",") -> tuple[list[list[float]], list[list[float]]]:
        """
        Lê arquivos CSV e separa as colunas de entrada (X) e saídas/targets (Y).
        Assume que as ÚLTIMAS 'num_target_columns' são os targets.
        """
        inputs = []
        targets = []

        with open(filepath, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:

                # pula linhas vazias/comentários
                if not row or row[0].startswith("#"):
                    continue

                float_row = [float(val) for val in row]
                X = float_row[:-num_target_columns]
                Y = float_row[-num_target_columns:]

                inputs.append(X)
                targets.append(Y)

        return inputs, targets

    @staticmethod
    def load_npy(x_path: str, y_path: str) -> tuple[list[list[float]], list[list[float]]]:
        """Lê os arquivos .npy fornecidos pelo NumPy e converte para listas de float puro."""
        x_data = np.load(x_path)
        y_data = np.load(y_path)

        # Garante que Y tenha 2D (ex: [N, 1] em vez de [N,])
        if len(y_data.shape) == 1:
            y_data = y_data.reshape(-1, 1)

        inputs = x_data.astype(float).tolist()
        targets = y_data.astype(float).tolist()

        return inputs, targets

    @staticmethod
    def load_caracteres_completo(
        filepath_inputs: str = "data/CARACTERES COMPLETO/X.txt",
        filepath_labels: str = "data/CARACTERES COMPLETO/Y_letra.txt",
        num_test_samples: int = 130,
    ) -> tuple[
        list[list[float]],  # train_inputs
        list[list[float]],  # train_targets
        list[list[float]],  # test_inputs
        list[list[float]],  # test_targets
        list[str],          # class_labels
    ]:
        """
        Carrega o dataset oficial CARACTERES COMPLETO (1326 amostras x 120 pixels).
        
        - Trata o arquivo X.txt descartando a 121ª coluna (gerada por vírgula residual ao final de cada linha).
        - Converte cada letra de Y_letra.txt em um vetor One-Hot Bipolar de 26 posições (-1.0 e +1.0).
        - Separa estritamente as últimas 'num_test_samples' (padrão 130) para teste,
            restando as 1196 primeiras amostras para treinamento.
        """
        # Lê as entradas descartando a vírgula residual no fim de cada linha (mantém 120 pixels)
        with open(filepath_inputs, "r", encoding="utf-8-sig") as f:
            all_inputs = [
                [float(pixel) for pixel in line.strip().split(",")[:120]]
                for line in f
                if line.strip()
            ]

        # Lê os rótulos de cada caractere ('A', 'B', ...)
        with open(filepath_labels, "r", encoding="utf-8-sig") as f:
            all_labels = [line.strip() for line in f if line.strip()]

        # Cria lista de classes de 'A' a 'Z' e mapeamento de índice
        class_labels = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
        label_to_index = {label: index for index, label in enumerate(class_labels)}

        # Converte cada rótulo em vetor One-Hot Bipolar (todas -1.0, alvo +1.0)
        all_targets: list[list[float]] = []
        for label in all_labels:
            target_vector = [-1.0] * len(class_labels)
            if label in label_to_index:
                target_vector[label_to_index[label]] = 1.0
            all_targets.append(target_vector)

        # Separação formal entre Treino e Teste
        num_train_samples = len(all_inputs) - num_test_samples

        train_inputs = all_inputs[:num_train_samples]
        train_targets = all_targets[:num_train_samples]

        test_inputs = all_inputs[num_train_samples:]
        test_targets = all_targets[num_train_samples:]

        return train_inputs, train_targets, test_inputs, test_targets, class_labels
