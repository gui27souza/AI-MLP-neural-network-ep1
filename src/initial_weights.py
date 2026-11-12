import numpy as np

DEFAULT_SEED = 42


def get_initial_weights(
    hidden_neurons: int,
    num_inputs: int = 120,
    num_outputs: int = 26,
    seed: int = DEFAULT_SEED,
    weight_range: tuple[float, float] = (-0.2, 0.2),
) -> dict:
    """
    Gera deterministicamente os pesos iniciais para uma dada arquitetura
    usando um gerador isolado com seed fixa.

    Garante que:
    1. Toda rodada com a mesma quantidade de neurônios ocultos parta EXATAMENTE do mesmo ponto.
    2. Comparações entre taxas de aprendizado e momentos ocorram sob a mesma superfície de partida.
    3. Nenhum efeito colateral de outras chamadas randômicas no código altere os pesos gerados.
    """
    rng = np.random.RandomState(seed + hidden_neurons)
    w_min, w_max = weight_range

    # Camada Oculta: shape (num_neurons, num_inputs)
    h_bias = rng.uniform(w_min, w_max, size=hidden_neurons).tolist()
    h_weights = rng.uniform(w_min, w_max, size=(hidden_neurons, num_inputs)).tolist()

    # Camada de Saída: shape (num_outputs, hidden_neurons)
    out_bias = rng.uniform(w_min, w_max, size=num_outputs).tolist()
    out_weights = rng.uniform(w_min, w_max, size=(num_outputs, hidden_neurons)).tolist()

    return {
        "hidden_layers": [
            {
                "bias": h_bias,
                "weights": h_weights,
            }
        ],
        "output_layer": {
            "bias": out_bias,
            "weights": out_weights,
        },
    }


# Alias para compatibilidade retroativa
get_grid_search_initial_weights = get_initial_weights
