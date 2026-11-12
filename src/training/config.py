from dataclasses import dataclass
import os
from typing import Callable, TypeVar

from src.data.data_augmentation import augment_bipolar_dataset

T = TypeVar("T")


# =========================================================
# Parsers Utilitários de Variáveis de Ambiente (Fail Fast)
# =========================================================

def _env(key: str, cast: Callable[[str], T], default: T | None = None, required: bool = False) -> T | None:
    val = os.getenv(key)
    if val is None or not val.strip():
        if required:
            raise KeyError(f"Variável de ambiente obrigatória '{key}' não foi informada.")
        return default
    return cast(val.strip())


def _env_list(key: str, cast: Callable[[str], T], default: list[T] | None = None, required: bool = False) -> list[T]:
    val = os.getenv(key)
    if val is None or not val.strip():
        if required:
            raise KeyError(f"Variável de ambiente obrigatória '{key}' não foi informada.")
        return default if default is not None else []
    return [cast(x.strip()) for x in val.split(",") if x.strip()]


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    return val.strip().lower() in ("1", "true", "yes") if val and val.strip() else default


def _env_seed(key: str = "SEED", default: int | None = 42) -> int | None:
    val = os.getenv(key)
    if not val or not val.strip():
        return default
    return None if val.strip().lower() in ("none", "null") else int(val.strip())


# ============================
# Dataclasses de Configuração
# ============================

@dataclass(frozen=True)
class AugmentConfig:
    """Configuração de Data Augmentation Bipolar."""
    noise_rate: float = 0.0
    copies: int = 1

    @property
    def enabled(self) -> bool:
        return self.noise_rate > 0.0

    @classmethod
    def from_env(cls) -> "AugmentConfig":
        return cls(
            noise_rate=_env("NOISE_RATE", float, default=0.0),
            copies=_env("AUGMENT_COPIES", int, default=1),
        )

    def apply(
        self,
        train_x: list[list[float]],
        train_y: list[list[float]],
        seed: int = 42,
    ) -> tuple[list[list[float]], list[list[float]]]:
        if not self.enabled:
            return train_x, train_y
        return augment_bipolar_dataset(
            train_x, train_y, noise_rate=self.noise_rate, copies_per_sample=self.copies, seed=seed
        )


@dataclass(frozen=True)
class TrainConfig:
    """Configuração de treino individual (src/main.py)."""
    epochs: int
    learn_rate: float
    hidden_neurons: int
    momentum: float
    val_ratio: float
    print_interval: int
    target_mse: float | None
    patience: int | None
    seed: int | None
    augment: AugmentConfig

    @classmethod
    def from_env(cls) -> "TrainConfig":
        return cls(
            epochs=_env("NUM_EPOCHS", int, required=True),
            learn_rate=_env("LEARN_RATE", float, required=True),
            hidden_neurons=_env("HIDDEN_NEURONS", int, required=True),
            momentum=_env("MOMENTUM", float, required=True),
            val_ratio=_env("VAL_RATIO", float, required=True),
            print_interval=_env("PRINT_INTERVAL", int, default=5),
            target_mse=_env("TARGET_MSE", float, default=None),
            patience=_env("PATIENCE", int, default=None),
            seed=_env_seed("SEED", default=42),
            augment=AugmentConfig.from_env(),
        )


@dataclass(frozen=True)
class GridSearchConfig:
    """Configuração para exploração em grade (src/experiments/grid_search.py)."""
    max_epochs: int
    val_ratios: list[float]
    hidden_neurons: list[int]
    learn_rates: list[float]
    momentums: list[float]
    patiences: list[int]
    noise_rates: list[float]
    target_mse: float | None
    min_acc_threshold: float
    seed: int | None
    augment_copies: int = 1

    @classmethod
    def from_env(cls) -> "GridSearchConfig":
        noise_rates = _env_list("NOISE_RATES", float, default=None)
        if noise_rates is None or len(noise_rates) == 0:
            single_noise = _env("NOISE_RATE", float, default=0.0)
            noise_rates = [single_noise]

        return cls(
            max_epochs=_env("MAX_EPOCHS", int, required=True),
            val_ratios=_env_list("VAL_RATIOS", float, required=True),
            hidden_neurons=_env_list("HIDDEN_NEURONS", int, required=True),
            learn_rates=_env_list("LEARN_RATE", float, required=True),
            momentums=_env_list("MOMENTUM", float, required=True),
            patiences=_env_list("PATIENCE", int, required=True),
            noise_rates=noise_rates,
            target_mse=_env("TARGET_MSE", float, default=None),
            min_acc_threshold=_env("MIN_ACC_THRESHOLD", float, default=75.0),
            seed=_env_seed("SEED", default=42),
            augment_copies=_env("AUGMENT_COPIES", int, default=1),
        )
