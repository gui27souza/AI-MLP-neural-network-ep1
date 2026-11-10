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
VAL_RATIO ?= 0.1
MODEL ?=

.PHONY: help main lab_logic_gates grid_search evaluate

# ==============================================================================
#  Execução
# ==============================================================================

help:
	@echo
	@echo "== AI MLP Neural Network =="
	@echo
	@echo "====> main -> executa o treinamento no CARACTERES COMPLETO"
	@echo "      Ex: make main NUM_EPOCHS=50 LEARN_RATE=0.02 HIDDEN_NEURONS=64 MOMENTUM=0.7 PRINT_INTERVAL=1 VAL_RATIO=0.2"
	@echo
	@echo "====> lab_logic_gates -> executa as portas lógicas"
	@echo "      Ex: make lab_logic_gates NUM_EPOCHS=500"
	@echo
	@echo "====> grid_search -> executa a busca em grade de hiperparâmetros"
	@echo "      Ex: make grid_search"
	@echo
	@echo "====> evaluate -> avalia um modelo salvo (matriz de confusao e curvas)"
	@echo "      Ex: make evaluate MODEL=models/grid_search/h80_acc86.4_.../gs_h80_...json"
	@echo

main:
	@NUM_EPOCHS=$(NUM_EPOCHS) \
	LEARN_RATE=$(LEARN_RATE) \
	HIDDEN_NEURONS=$(HIDDEN_NEURONS) \
	MOMENTUM=$(MOMENTUM) \
	TARGET_MSE=$(TARGET_MSE) \
	PATIENCE=$(PATIENCE) \
	PRINT_INTERVAL=$(PRINT_INTERVAL) \
	VAL_RATIO=$(VAL_RATIO) \
	python3 -m src.main

lab_logic_gates:
	@NUM_EPOCHS=$(NUM_EPOCHS) \
	python3 -m lab.logic_gates

grid_search:
	@python3 -m lab.grid_search.grid_search

evaluate:
	@MODEL="$(MODEL)" python3 -m lab.evaluator
