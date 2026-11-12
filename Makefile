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
SEED ?= 42
AUGMENT_DATA ?= 0
NOISE_RATE ?= 0.05
AUGMENT_COPIES ?= 1
MODEL ?=

.PHONY: help main lab_logic_gates grid_search evaluate full_pipeline

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
	@echo "====> full_pipeline -> executa treino completo (main) e avaliacao automatica (evaluate)"
	@echo "      Ex: make full_pipeline NUM_EPOCHS=120 LEARN_RATE=0.010 HIDDEN_NEURONS=48 MOMENTUM=0.55 PATIENCE=6 VAL_RATIO=0.3"
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
	SEED=$(SEED) \
	AUGMENT_DATA=$(AUGMENT_DATA) \
	NOISE_RATE=$(NOISE_RATE) \
	AUGMENT_COPIES=$(AUGMENT_COPIES) \
	python3 -m src.main

lab_logic_gates:
	@NUM_EPOCHS=$(NUM_EPOCHS) \
	python3 -m lab.logic_gates

grid_search:
	@python3 -m lab.grid_search.grid_search

evaluate:
	@MODEL="$(MODEL)" python3 -m lab.evaluator

full_pipeline: main
	@if [ -f "models/.latest_model.txt" ]; then \
		LATEST_MODEL=$$(cat models/.latest_model.txt); \
		echo ""; \
		echo "== Iniciando avaliacao automatica do modelo gerado: $$LATEST_MODEL =="; \
		echo ""; \
		MODEL="$$LATEST_MODEL" python3 -m lab.evaluator; \
	else \
		echo "Aviso: Nenhum modelo foi registrado para avaliacao."; \
	fi
