"""
Verificação oficial do Teste de Mesa da disciplina (docs/exemplo-numerico-MLP.pdf).

Reproduz exatamente o Exemplo Numérico da Seção 3 do documento de apoio dos professores
André Paulino de Lima e Sarajane Marques Peres (EACH-USP), validando cada passo
modular da implementação de `Layer` e `MLP`:
  - Seção 3.1: Propagação direta (Feedforward) -> saídas z da camada escondida e y da saída
  - Seção 3.2: Termo de erro δ_k e ajustes ΔW na camada de saída
  - Seção 3.3: Contribuição de erro δ_j e ajustes ΔV na camada escondida
  - Seção 3.4: Novos pesos atualizados nas matrizes W e V
"""

from src.core.funcs import deriv_sigmoide_binaria, sigmoide_binaria
from src.core.models import Layer, MLP
from src.persistence.mlp_persister import MLPPersister


def run_mesa_test() -> None:
    print("=" * 76)
    print("  VERIFICAÇÃO DO TESTE DE MESA OFICIAL (docs/exemplo-numerico-MLP.pdf)")
    print("=" * 76)

    # Topologia do exemplo numérico (Seção 3, Figura 3):
    # n = 2 entradas, p = 3 neurônios na camada escondida, m = 2 neurônios na saída
    # Função de ativação: sigmoide binária f(x) = 1 / (1 + e^-x), taxa α = 0.5
    hidden_layer = Layer(
        num_inputs=2,
        num_neurons=3,
        activation_func=sigmoide_binaria,
        derivative_activation=deriv_sigmoide_binaria,
    )

    mlp = MLP(
        num_inputs=2,
        num_outputs=2,
        learn_rate=0.5,
        hidden_layers=[hidden_layer],
        output_activation=sigmoide_binaria,
        output_deriv_activation=deriv_sigmoide_binaria,
        momentum=0.0,
    )

    # Pesos iniciais V e W definidos na página 6 do PDF (coluna 0 = bias):
    # V (3 x 3):
    #   [-0.1,  0.1, -0.1]
    #   [-0.1,  0.1,  0.1]
    #   [ 0.1, -0.1, -0.1]
    # W (2 x 4):
    #   [-0.1,  0.1,  0.0,  0.1]
    #   [ 0.1, -0.1,  0.1, -0.1]
    initial_state = MLPPersister.from_fausett_matrices(
        hidden_matrices=[
            [
                [-0.1, 0.1, -0.1],
                [-0.1, 0.1, 0.1],
                [0.1, -0.1, -0.1],
            ]
        ],
        output_matrix=[
            [-0.1, 0.1, 0.0, 0.1],
            [0.1, -0.1, 0.1, -0.1],
        ],
    )
    mlp.set_initial_weights(initial_state)

    x_sample = [1.0, 1.0]
    t_sample = [1.0, 0.0]

    # 1. Passo Forward (Seção 3.1)
    y_out = mlp.predict(x_sample)
    z_out = mlp.hidden_layers[0].last_outputs

    expected_z = [0.4750, 0.5250, 0.4750]
    expected_y = [0.4988, 0.5144]

    print("\n[Seção 3.1] Cálculo da saída da rede neural (Forward Step):")
    print(f"  z calculado (camada escondida) : {[round(v, 4) for v in z_out]}")
    print(f"  z esperado no PDF              : {expected_z}")
    print(f"  y calculado (camada de saída)  : {[round(v, 4) for v in y_out]}")
    print(f"  y esperado no PDF              : {expected_y}")

    assert [round(v, 4) for v in z_out] == expected_z, "Divergência em z!"
    assert [round(v, 4) for v in y_out] == expected_y, "Divergência em y!"

    # 2. Passo Backward — Deltas (Seções 3.2 e 3.3)
    delta_out = mlp._compute_output_deltas(t_sample, y_out)
    delta_hid = mlp._compute_hidden_deltas(delta_out)[0]

    expected_delta_out = [0.1253, -0.1285]
    expected_delta_hid = [0.0063, -0.0032, 0.0063]

    print("\n[Seção 3.2] Erro retropropagado na camada de saída (δ_k):")
    print(f"  δ_k calculado : {[round(v, 4) for v in delta_out]}")
    print(f"  δ_k esperado  : {expected_delta_out}")

    print("\n[Seção 3.3] Contribuição do erro na camada escondida (δ_j):")
    print(f"  δ_j calculado : {[round(v, 4) for v in delta_hid]}")
    print(f"  δ_j esperado  : {expected_delta_hid}")

    assert [round(v, 4) for v in delta_out] == expected_delta_out, "Divergência em δ_k!"
    assert [round(v, 4) for v in delta_hid] == expected_delta_hid, "Divergência em δ_j!"

    # 3. Atualização dos Pesos (Seção 3.4)
    mlp._apply_weight_updates(delta_out, [delta_hid])
    v_mats, w_mat = MLPPersister.to_fausett_matrices(mlp.get_weights())
    v_updated = [[round(float(val), 4) for val in row] for row in v_mats[0]]
    w_updated = [[round(float(val), 4) for val in row] for row in w_mat]

    expected_w_updated = [
        [-0.0373, 0.1298, 0.0329, 0.1298],
        [0.0358, -0.1305, 0.0663, -0.1305],
    ]
    expected_v_updated = [
        [-0.0968, 0.1032, -0.0968],
        [-0.1016, 0.0984, 0.0984],
        [0.1032, -0.0968, -0.0968],
    ]

    print("\n[Seção 3.4] Ajuste dos Pesos (Matrizes W e V atualizadas):")
    print(f"  W atualizado calculado : {w_updated}")
    print(f"  W atualizado esperado  : {expected_w_updated}")
    print(f"  V atualizado calculado : {v_updated}")
    print(f"  V atualizado esperado  : {expected_v_updated}")

    assert w_updated == expected_w_updated, "Divergência na matriz W atualizada!"
    assert v_updated == expected_v_updated, "Divergência na matriz V atualizada!"

    print("\n" + "=" * 76)
    print("  SUCESSO: Todos os valores bateram 100% (4 casas decimais) com o PDF!")
    print("=" * 76)


if __name__ == "__main__":
    run_mesa_test()
