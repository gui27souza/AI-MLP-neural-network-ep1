#!/usr/bin/env bash
# =====================================
#  Script de Execução para Grid Search
# =====================================

# Hiperparâmetros Fixos e Critérios
export MAX_EPOCHS=120
export TARGET_MSE=0.001
export MIN_ACC_THRESHOLD=75.0
export SEED=42

# Data Augmentation Bipolar (0.0 = limpo, ex: "0.0,0.03" para comparar com/sem ruido)
export NOISE_RATES="0.0"
export AUGMENT_COPIES=1

# Grade de Busca (valores separados por vírgula)
export VAL_RATIOS="0.30"
export HIDDEN_NEURONS="16,48,64"
export LEARN_RATE="0.008,0.010,0.025"
export MOMENTUM="0.15,0.55,0.85"
export PATIENCE="6"

# Executa o grid search e gera automaticamente o relatório e gráficos
make full_grid_pipeline
