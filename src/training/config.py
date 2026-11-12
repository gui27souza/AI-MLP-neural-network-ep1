import os
from dataclasses import dataclass

from src.data.data_augmentation import augment_bipolar_dataset


def env_bool(key: str, default: bool = False) -> bool:
    """Converte valor de variavel de ambiente para booleano."""
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default
    return val.strip().lower() in ("1", "true", "yes")


def env_int(key: str, default: int | None = None, required: bool = False) -> int | None:
    """Converte valor de variavel de ambiente para inteiro."""
    val = os.getenv(key)
    if val is None or val.strip() == "":
        if required:
            raise KeyError(f"Variavel de ambiente obrigatoria '{key}' nao foi informada.")
        return default
    return int(val.strip())


def env_float(key: str, default: float | None = None, required: bool = False) -> float | None:
    """Converte valor de variavel de ambiente para ponto flutuante."""
    val = os.getenv(key)
    if val is None or val.strip() == "":
        if required:
            raise KeyError(f"Variavel de ambiente obrigatoria '{key}' nao foi informada.")
        return default
    return float(val.strip())


def env_list(key: str, default: list, cast: type) -> list:
    """Converte string separada por virgula em lista tipada."""
    val = os.getenv(key)
    if not val:
        return default
    return [cast(x.strip()) for x in val.split(",") if x.strip()]


def env_seed(key: str = "SEED", default: int | None = 42) -> int | None:
    """Converte variavel de seed para inteiro ou None se desativada."""
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default
    clean = val.strip().lower()
    if clean in ("none", "null"):
        return None
    return int(clean)


@dataclass(frozen=True)
class AugmentConfig:
    """Configuracao de Data Augmentation Bipolar."""
    enabled: bool = False
    noise_rate: float = 0.05
    copies: int = 1

    @classmethod
    def from_env(cls) -> "AugmentConfig":
        return cls(
            enabled=env_bool("AUGMENT_DATA", default=False),
            noise_rate=env_float("NOISE_RATE", default=0.05),
            copies=env_int("AUGMENT_COPIES", default=1),
        )

    def apply(
        self,
        train_x: list[list[float]],
        train_y: list[list[float]],
        seed: int = 42,
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Aplica aumento de dados com ruido bipolar se habilitado."""
        if not self.enabled:
            return train_x, train_y
        return augment_bipolar_dataset(
            train_x,
            train_y,
            noise_rate=self.noise_rate,
            copies_per_sample=self.copies,
            seed=seed,
        )


@dataclass(frozen=True)
class TrainConfig:
    """Configuracao de treino individual (src/main.py)."""
    epochs: int
    learn_rate: float
    hidden_neurons: int
    momentum: float
    print_interval: int
    val_ratio: float
    target_mse: float | None
    patience: int | None
    seed: int | None
    augment: AugmentConfig

    @classmethod
    def from_env(cls) -> "TrainConfig":
        return cls(
            epochs=env_int("NUM_EPOCHS", required=True),
            learn_rate=env_float("LEARN_RATE", required=True),
            hidden_neurons=env_int("HIDDEN_NEURONS", required=True),
            momentum=env_float("MOMENTUM", required=True),
            print_interval=env_int("PRINT_INTERVAL", default=5),
            val_ratio=env_float("VAL_RATIO", default=0.3),
            target_mse=env_float("TARGET_MSE", default=None),
            patience=env_int("PATIENCE", default=None),
            seed=env_seed("SEED", default=42),
            augment=AugmentConfig.from_env(),
        )


@dataclass(frozen=True)
class GridSearchConfig:
    """Configuracao para exploracao em grade (src/grid_search.py)."""
    max_epochs: int
    target_mse: float | None
    min_acc_threshold: float
    seed: int | None
    val_ratios: list[float]
    hidden_neurons: list[int]
    learn_rates: list[float]
    momentums: list[float]
    patiences: list[int]
    augment: AugmentConfig

    @classmethod
    def from_env(cls) -> "GridSearchConfig":
        lr_key = "LEARNING_RATES" if "LEARNING_RATES" in os.environ else "LEARN_RATE"
        m_key = "MOMENTUMS" if "MOMENTUMS" in os.environ else "MOMENTUM"
        epochs = env_int("MAX_EPOCHS")
        if epochs is None:
            epochs = env_int("NUM_EPOCHS", default=120)

        return cls(
            max_epochs=epochs,
            target_mse=env_float("TARGET_MSE", default=0.001),
            min_acc_threshold=env_float("MIN_ACC_THRESHOLD", default=75.0),
            seed=env_seed("SEED", default=42),
            val_ratios=env_list("VAL_RATIOS", [0.30], float),
            hidden_neurons=env_list("HIDDEN_NEURONS", [16, 48, 64], int),
            learn_rates=env_list(lr_key, [0.008, 0.010, 0.025], float),
            momentums=env_list(m_key, [0.15, 0.55, 0.85], float),
            patiences=env_list("PATIENCE", [6], int),
            augment=AugmentConfig.from_env(),
        )
