#!/usr/bin/env bash
# ==============================================================================================
#  GRID SEARCH AMPLO DETERMINÍSTICO
# ==============================================================================================
#
# Objetivo:
#   Mapeamento analítico e exaustivo do espaço de hiperparâmetros da MLP,
#   explorando contrastes teóricos em topologia, gradiente, momento e ruído.
#
# Dimensões da Grade:
#   - Val Ratios       : 0.15, 0.30, 0.45 (3 valores: validação enxuta -> holdout amplo)
#   - Hidden Neurons   : 24, 48, 96, 144 (4 valores: subcapacidade -> sobreparametrização)
#   - Learn Rate (eta) : 0.002, 0.020, 0.07, 0.1 (4 valores: rastejante -> agressivo)
#   - Momentum (alpha) : 0.0, 0.40, 0.70, 0.90 (4 valores: SGD puro -> inércia pesada)
#   - Patience         : 14 (1 valor: respiro estendido fixo)
#   - Noise Rates      : 0.0, 0.03, 0.08 (3 valores: limpo -> ruído leve -> ruído estresse)
#   Total              : 3 * 4 * 4 * 4 * 1 * 3 = 576 modelos
#
# ==============================================================================================

set -e

# == Hiperparâmetros Globais e Critérios de Parada ====
export MAX_EPOCHS=200
export TARGET_MSE=0.00001
export MIN_ACC_THRESHOLD=78.0
export SEED=42

# == Data Augmentation Bipolar (0.0 = limpo, > 0.0 = taxa estocástica) ====
export NOISE_RATES="0.0,0.03,0.08"
export AUGMENT_COPIES=1

# == Espaço de Hiperparâmetros (Combinatória Cartesiana) ====
export VAL_RATIOS="0.15,0.30,0.45"
export HIDDEN_NEURONS="24,48,96,144"
export LEARN_RATE="0.002,0.020,0.07,0.1"
export MOMENTUM="0.0,0.40,0.70,0.90"
export PATIENCE="14"

# == Executa a Grade e Gera Relatórios + Gráficos Automaticamente ====
echo "=============================================================================="
echo "  Iniciando Mega Grid Search Amplo (576 combinacoes)"
echo "  Data/Hora: $(date)"
echo "  Semente: ${SEED} | Epocas Max: ${MAX_EPOCHS} | Target MSE: ${TARGET_MSE}"
echo "=============================================================================="

make full_grid_pipeline