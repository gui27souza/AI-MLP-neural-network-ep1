# ==============================================================================
#  Configuração do Ambiente
# ==============================================================================

PYTHON ?= $(if $(wildcard venv/bin/python),venv/bin/python,python3)

.PHONY: help setup main lab_logic_gates grid_search evaluate evaluate_grid full_pipeline full_grid_pipeline

# ==============================================================================
#  Ajuda e Setup
# ==============================================================================

help:
	@echo
	@echo "== AI MLP Neural Network =="
	@echo
	@echo "====> setup -> cria os scripts locais run.sh e run_grid.sh a partir dos templates"
	@echo "      Ex: make setup"
	@echo
	@echo "====> main -> executa o treinamento no CARACTERES COMPLETO"
	@echo "      Ex: make main NUM_EPOCHS=50 LEARN_RATE=0.02 HIDDEN_NEURONS=64 MOMENTUM=0.7 VAL_RATIO=0.3"
	@echo
	@echo "====> lab_logic_gates -> executa as portas lógicas"
	@echo "      Ex: make lab_logic_gates NUM_EPOCHS=500"
	@echo
	@echo "====> grid_search -> executa a busca em grade de hiperparâmetros"
	@echo "      Ex: ./run_grid.sh ou make grid_search MAX_EPOCHS=120 HIDDEN_NEURONS=16,48 LEARN_RATE=0.01 MOMENTUM=0.5 PATIENCE=6 VAL_RATIOS=0.3"
	@echo
	@echo "====> evaluate -> avalia um modelo salvo (matriz de confusao e curvas)"
	@echo "      Ex: make evaluate MODEL=models/grid_search/h80_acc86.4_.../gs_h80_...json"
	@echo
	@echo "====> full_pipeline -> executa treino completo (main) e avaliacao automatica (evaluate)"
	@echo "      Ex: ./run.sh"
	@echo

setup:
	@if [ ! -f run.sh ]; then cp run.example.sh run.sh && chmod +x run.sh && echo "Criado run.sh a partir do template."; else echo "run.sh ja existe."; fi
	@if [ ! -f run_grid.sh ]; then cp run_grid.example.sh run_grid.sh && chmod +x run_grid.sh && echo "Criado run_grid.sh a partir do template."; else echo "run_grid.sh ja existe."; fi

main:
	@NUM_EPOCHS="$(NUM_EPOCHS)" \
	LEARN_RATE="$(LEARN_RATE)" \
	HIDDEN_NEURONS="$(HIDDEN_NEURONS)" \
	MOMENTUM="$(MOMENTUM)" \
	VAL_RATIO="$(VAL_RATIO)" \
	TARGET_MSE="$(TARGET_MSE)" \
	PATIENCE="$(PATIENCE)" \
	PRINT_INTERVAL="$(PRINT_INTERVAL)" \
	SEED="$(SEED)" \
	AUGMENT_DATA="$(AUGMENT_DATA)" \
	NOISE_RATE="$(NOISE_RATE)" \
	AUGMENT_COPIES="$(AUGMENT_COPIES)" \
	$(PYTHON) -m src.main

lab_logic_gates:
	@NUM_EPOCHS="$(or $(NUM_EPOCHS),1000)" \
	$(PYTHON) -m lab.logic_gates.logic_gates

grid_search:
	@MAX_EPOCHS="$(MAX_EPOCHS)" \
	VAL_RATIOS="$(VAL_RATIOS)" \
	HIDDEN_NEURONS="$(HIDDEN_NEURONS)" \
	LEARN_RATE="$(LEARN_RATE)" \
	MOMENTUM="$(MOMENTUM)" \
	PATIENCE="$(PATIENCE)" \
	TARGET_MSE="$(TARGET_MSE)" \
	MIN_ACC_THRESHOLD="$(MIN_ACC_THRESHOLD)" \
	SEED="$(SEED)" \
	AUGMENT_DATA="$(AUGMENT_DATA)" \
	NOISE_RATE="$(NOISE_RATE)" \
	AUGMENT_COPIES="$(AUGMENT_COPIES)" \
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
