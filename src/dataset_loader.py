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