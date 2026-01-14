import torch
import time
from resonator_ml.machine_learning.loop_filter.neural_network import Trainer
def print_callback(epoch: int, epochs: int, epoch_loss: float):
    print(f"Epoch {epoch + 1}/{epochs}  Loss: {epoch_loss:.10f}")


class TrainLoopNetwork:
    def __init__(self, model, training_data_generator, model_file_name, trainer: Trainer):
        self.model = model
        self.training_data_generator = training_data_generator
        self.model_file_name = model_file_name
        self.trainer = trainer

    def execute(self):

        start = time.time()

        dataloader = self.training_data_generator.generate_training_dataloader()


        model = self.trainer.train_neural_network(self.model, dataloader, epoch_callback=print_callback)

        # Speichern
        torch.save(model.state_dict(), self.model_file_name)

        print("Training abgeschlossen.")
        end = time.time()
        length = end - start
        # Show the results : this can be altered however you like
        print("It took", length, "seconds!")