.PHONY: run_mlp

EPOCHS ?= 1000

help:
	@echo
	@echo "== AI MLP Neural Network"
	@echo
	@echo "====> lab_logic_gates -> executa as redes referentes às portas lógicas AND, OR e XOR"
	@echo

lab_logic_gates:
	@NUM_EPOCHS=$(EPOCHS) \
	python3 -m lab.logic_gates
