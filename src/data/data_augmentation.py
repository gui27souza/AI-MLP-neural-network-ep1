import random
from typing import Sequence

DEFAULT_AUGMENTATION_SEED = 42


def add_bipolar_noise(
    vector: Sequence[float],
    noise_rate: float = 0.05,
    rng: random.Random | None = None,
) -> list[float]:
    """Inverte pixels bipolares (-1.0 <-> +1.0) com probabilidade noise_rate."""
    if rng is None:
        rng = random.Random()
    return [-p if rng.random() < noise_rate else p for p in vector]


def augment_bipolar_dataset(
    inputs: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    noise_rate: float = 0.05,
    copies_per_sample: int = 1,
    keep_original: bool = True,
    seed: int = DEFAULT_AUGMENTATION_SEED,
) -> tuple[list[list[float]], list[list[float]]]:
    """
    Aumenta deterministicamente o treino gerando copias com inversao bipolar de pixels.
    NUNCA deve ser aplicado ao conjunto de teste/validacao.
    """
    if len(inputs) != len(targets):
        raise ValueError("Tamanho incompativel entre inputs e targets.")

    rng = random.Random(seed)
    augmented_inputs = [list(x) for x in inputs] if keep_original else []
    augmented_targets = [list(y) for y in targets] if keep_original else []

    for _ in range(copies_per_sample):
        for x, y in zip(inputs, targets):
            augmented_inputs.append(add_bipolar_noise(x, noise_rate=noise_rate, rng=rng))
            augmented_targets.append(list(y))

    return augmented_inputs, augmented_targets
