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
PYTHON ?= $(if $(wildcard venv/bin/python),venv/bin/python,python3)

.PHONY: help main lab_logic_gates grid_search evaluate evaluate_grid full_pipeline full_grid_pipeline

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
	$(PYTHON) -m src.main

lab_logic_gates:
	@NUM_EPOCHS=$(NUM_EPOCHS) \
	$(PYTHON) -m lab.logic_gates.logic_gates

grid_search:
	@MAX_EPOCHS="$(MAX_EPOCHS)" \
	TARGET_MSE="$(TARGET_MSE)" \
	MIN_ACC_THRESHOLD="$(MIN_ACC_THRESHOLD)" \
	SEED="$(SEED)" \
	AUGMENT_DATA="$(AUGMENT_DATA)" \
	NOISE_RATE="$(NOISE_RATE)" \
	AUGMENT_COPIES="$(AUGMENT_COPIES)" \
	VAL_RATIOS="$(VAL_RATIOS)" \
	HIDDEN_NEURONS="$(HIDDEN_NEURONS)" \
	LEARN_RATE="$(LEARN_RATE)" \
	MOMENTUM="$(MOMENTUM)" \
	PATIENCE="$(PATIENCE)" \
	$(PYTHON) -m src.experiments.grid_search

evaluate:
	@MODEL="$(MODEL)" $(PYTHON) -m lab.evaluation.evaluator

evaluate_grid:
	@GRID_DIR="$(or $(GRID_DIR),$(DIR))" $(PYTHON) -m lab.analysis.grid_evaluator

full_grid_pipeline: grid_search
	@if [ -f "models/grid_search/.latest_grid.txt" ]; then \
		LATEST_GRID=$$(cat models/grid_search/.latest_grid.txt); \
		echo ""; \
		echo "== Executando analise e geracao de graficos do Grid Search: $$LATEST_GRID =="; \
		echo ""; \
		$(PYTHON) -m lab.analysis.grid_evaluator "$$LATEST_GRID"; \
	else \
		echo "Aviso: Nenhum diretorio de grid search foi registrado."; \
	fi

full_pipeline: main
	@if [ -f "models/.latest_model.txt" ]; then \
		LATEST_MODEL=$$(cat models/.latest_model.txt); \
		echo ""; \
		echo "== Iniciando avaliacao automatica do modelo gerado: $$LATEST_MODEL =="; \
		echo ""; \
		MODEL="$$LATEST_MODEL" $(PYTHON) -m lab.evaluation.evaluator; \
	else \
		echo "Aviso: Nenhum modelo foi registrado para avaliacao."; \
	fi
