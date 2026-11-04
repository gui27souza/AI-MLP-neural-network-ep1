from datetime import (
    datetime, timedelta
)

from .models import MLP


class Orchestrator:

    start_ts: float
    end_ts: float
    delta_ts: float
    duration: timedelta

    def __init__(
        self,
        mlp: MLP, epochs: int,
        inputs: list[list[float]],
        targets: list[list[float]],
    ):

        self.mlp = mlp
        self.epochs = epochs
        self.inputs = inputs
        self.targets = targets

        self.total_epochs = epochs

    def run(self, print_epochs_interval: int = 100):

        self.start_ts = datetime.now().timestamp()

        i = 1
        for i in range(1, self.epochs+1):

            print_now = (i % print_epochs_interval == 0)

            if print_now: print(f"\n=== Epoch {i}")

            self.mlp.run_epoch(
                inputs_list=self.inputs,
                targets=self.targets,
                print_info=print_now,
            )

            if print_now: print("==========")

        self.end_ts = datetime.now().timestamp()
        self.total_epochs = i

        self.delta_ts = self.end_ts - self.start_ts
        self.duration = timedelta(seconds=self.delta_ts)

    def print_results(self):
        print(f"Tempo total de execução: {self.delta_ts:.2f}s ({self.duration})")
