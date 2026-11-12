#!/usr/bin/env bash
# ================================
#  Script de Execução para Modelo
# ================================

# Configuração de Execução
export SEED=42
export PRINT_INTERVAL=4

# Hiperparâmetros Fixos e Critérios
export VAL_RATIO=0.30
export HIDDEN_NEURONS=48
export NUM_EPOCHS=120
export LEARN_RATE=0.025
export MOMENTUM=0.55
export PATIENCE=6
export TARGET_MSE=0.001

# Data Augmentation Bipolar (0.0 = desativado/limpo, > 0.0 = taxa de inversao)
export NOISE_RATE=0.0
export AUGMENT_COPIES=1

# ===

# Executa o treinamento de modelo e
# gera automaticamente o relatório e gráficos
make full_pipeline
