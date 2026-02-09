import torch
import time

from app.config.app import Config
from resonator_ml.machine_learning.loop_filter.neural_network import Trainer, NeuralNetworkModule
from resonator_ml.machine_learning.loop_filter.training_data import TrainingDataGenerator
from resonator_ml.ports.file_storage import FileStorage, DictStorage
import torch.nn as nn

def print_callback(epoch: int, epochs: int, epoch_loss: float, min_batch_loss: float, max_batch_loss: float):
    print(f"Epoch {epoch + 1}/{epochs}  Loss: {epoch_loss:.10f} Min_Batch_Loss: {min_batch_loss:.10f} Max_Batch_Loss: {max_batch_loss:.10f}")


class TrainLoopNetwork:
    # TODO model is only concrete NeuralNetworkModule bc of param logging. Clean up parameter logging, then make model
    # back nn.Module
    def __init__(self, model: NeuralNetworkModule, training_data_generator: TrainingDataGenerator, file_storage: FileStorage, trainer: Trainer, params_storage: DictStorage, app_config: Config):
        self.model = model
        self.training_data_generator = training_data_generator
        self.file_storage = file_storage
        self.trainer = trainer
        self.params_storage = params_storage
        self.app_config = app_config

    def execute(self):

        start = time.time()

        dataloader = self.training_data_generator.generate_training_dataloader()
        dataloader_time = time.time()
        print("Dataloader time ", dataloader_time - start, "seconds!")

        params = {
            'instrument': self.app_config.instrument_name,
            'batch_size': self.training_data_generator.training_parameters.batch_size,
            'lr': self.training_data_generator.training_parameters.learning_rate,
            'max_num_frames': self.training_data_generator.training_parameters.max_training_data_frames,
            'num_layers': len(self.model.common_net) + 1,
            'num_hidden': self.model.hidden
        }
        self.params_storage.save_dict(params)

        if self.file_storage.model_file_path().exists():
            self.model.load_state_dict(torch.load(self.file_storage.model_file_path()))

        model = self.trainer.train_neural_network(self.model, dataloader, epoch_callback=print_callback)


        # Save
        # TODO once logging is cleaned up, this is the right spot to make the new version
        # self.file_storage.make_new_version_output_dir() # Training the network means new version
        # Saving now in train
        # torch.save(model.state_dict(), self.file_storage.model_file_path())

        print("Training abgeschlossen.")
        end = time.time()
        length = end - start
        # Show the results : this can be altered however you like
        print("It took", length, "seconds!")