# ==============================================================================
#  Configuração de Variáveis (com defaults)
# ==============================================================================

NUM_EPOCHS     ?= 20
LEARN_RATE     ?= 0.015
HIDDEN_NEURONS ?= 32

.PHONY: help main lab_logic_gates

# ==============================================================================
#  Execução
# ==============================================================================

help:
	@echo
	@echo "== AI MLP Neural Network =="
	@echo
	@echo "====> main -> executa o treinamento no CARACTERES COMPLETO"
	@echo "      Ex: make main NUM_EPOCHS=50 LEARN_RATE=0.02 HIDDEN_NEURONS=64"
	@echo
	@echo "====> lab_logic_gates -> executa as portas lógicas"
	@echo "      Ex: make lab_logic_gates NUM_EPOCHS=500"
	@echo

main:
	@NUM_EPOCHS=$(NUM_EPOCHS) \
	LEARN_RATE=$(LEARN_RATE) \
	HIDDEN_NEURONS=$(HIDDEN_NEURONS) \
	python3 -m src.main

lab_logic_gates:
	@NUM_EPOCHS=$(NUM_EPOCHS) \
	python3 -m lab.logic_gates
