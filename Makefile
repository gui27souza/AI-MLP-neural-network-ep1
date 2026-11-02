.PHONY: run_mlp

EPOCHS ?= 1000

run_mlp:
	NUM_EPOCHS=$(EPOCHS) \
	python3 -m src.main
