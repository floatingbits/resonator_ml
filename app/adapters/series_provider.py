from networkx.classes import non_edges

from resonator_ml.ports.series_provider import SeriesProvider
import re
from pathlib import Path
from typing import List


def parse_training_logs_loss(log_text: str) -> List[float]:
    """Parse epoch number and loss from training logs."""
    results = []

    for line in log_text.strip().split('\n'):
        if 'Epoch' in line and 'Loss:' in line:
            # Split nach "Epoch " und dann nach "/"
            # epoch_part = line.split('Epoch ')[1].split('/')[0]
            # epoch = int(epoch_part)

            # Split nach "Loss: "
            loss_part = line.split('Loss: ')[1]
            loss = float(loss_part)

            results.append( loss)

    return results

def parse_training_logs_parameters(log_text: str) -> List[float]:
    """Parse epoch number and loss from training logs."""
    results = []

    for line in log_text.strip().split('\n'):
        if 'Epoch' in line and 'Loss:' in line:
            # Split nach "Epoch " und dann nach "/"
            # epoch_part = line.split('Epoch ')[1].split('/')[0]
            # epoch = int(epoch_part)

            # Split nach "Loss: "
            loss_part = line.split('Loss: ')[1]
            loss = float(loss_part)

            results.append( loss)

    return results

class TrainingLossSeriesProvider(SeriesProvider):
    def __init__(self, search_path: Path):
        self.search_path = search_path


    def files(self):
        return list(self.search_path.rglob("train_loop_network.log"))
    def data_at(self, index: int):
        filepath = self.files()[index]
        with open(filepath, 'r') as file:
            data = file.read()
            series = parse_training_logs_loss(data)

        return series
    def title_at(self, index: int) -> str:
        file = self.files()[index]
        params_file = list(file.parent.rglob("params.json"))
        if len(params_file) and params_file[0]:
            with params_file[0].open("r", encoding="utf-8") as f:
                data = f.read()
                return data.replace("\n", "").replace("{", "").replace("}", "")
        return file.as_posix()
    def num_plots(self) -> int:
        return len(self.files())