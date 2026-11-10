# ==============================================================================
#  Configuração de Variáveis (com defaults)
# ==============================================================================

NUM_EPOCHS     ?= 20
LEARN_RATE     ?= 0.015
HIDDEN_NEURONS ?= 32
MOMENTUM ?= 0.5
TARGET_MSE ?=
PATIENCE ?=
PRINT_INTERVAL ?= 2

.PHONY: help main lab_logic_gates

# ==============================================================================
#  Execução
# ==============================================================================

help:
	@echo
	@echo "== AI MLP Neural Network =="
	@echo
	@echo "====> main -> executa o treinamento no CARACTERES COMPLETO"
	@echo "      Ex: make main NUM_EPOCHS=50 LEARN_RATE=0.02 HIDDEN_NEURONS=64 MOMENTUM=0.7 PRINT_INTERVAL=1"
	@echo
	@echo "====> lab_logic_gates -> executa as portas lógicas"
	@echo "      Ex: make lab_logic_gates NUM_EPOCHS=500"
	@echo

main:
	@NUM_EPOCHS=$(NUM_EPOCHS) \
	LEARN_RATE=$(LEARN_RATE) \
	HIDDEN_NEURONS=$(HIDDEN_NEURONS) \
	MOMENTUM=$(MOMENTUM) \
	TARGET_MSE=$(TARGET_MSE) \
	PATIENCE=$(PATIENCE) \
	PRINT_INTERVAL=$(PRINT_INTERVAL) \
	python3 -m src.main

lab_logic_gates:
	@NUM_EPOCHS=$(NUM_EPOCHS) \
	python3 -m lab.logic_gates
