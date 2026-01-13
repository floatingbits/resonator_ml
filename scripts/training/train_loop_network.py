import torch
import time
from resonator_ml.machine_learning.file_management import create_loop_filter_model_file_name
from resonator_ml.machine_learning.loop_filter.neural_network import prepare_dataloader, NeuralNetworkModule, \
    train_neural_network, NeuralNetworkResonatorFactory
from resonator_ml.machine_learning.loop_filter.training_data import TrainingDataGenerator, TrainingParameterFactory


def print_callback(epoch: int, epochs: int, epoch_loss: float):
    print(f"Epoch {epoch + 1}/{epochs}  Loss: {epoch_loss:.10f}")

if __name__ == "__main__":
    instrument = "Strat_E"
    resonator_type_name = "v1"
    model_suffix = "_4"
    #model_suffix = "_test_v09"

    training_parameter_factory = TrainingParameterFactory()
    training_parameters = training_parameter_factory.create_parameters("v2.1")


    start = time.time()
    resonator_factory = NeuralNetworkResonatorFactory()

    training_data_generator = TrainingDataGenerator()
    dataset = training_data_generator.generate_training_dataset(network_type=resonator_type_name, instrument=instrument)

    dataloader = prepare_dataloader(dataset, batch_size=training_parameters.batch_size)


    model_name = instrument + "_" + resonator_type_name + model_suffix
    # Modell erstellen
    resonator = resonator_factory.create_neural_network_resonator(resonator_type_name, 44100)
    model = resonator.model

    # Trainieren
    model = train_neural_network(model, dataloader, lr=training_parameters.learning_rate,
                                 loss_fn=training_parameters.loss_function,epochs=training_parameters.epochs,
                                 epoch_callback=print_callback)

    # Speichern
    torch.save(model.state_dict(), create_loop_filter_model_file_name(model_name))

    print("Training abgeschlossen.")
    end = time.time()
    length = end - start
    # Show the results : this can be altered however you like
    print("It took", length, "seconds!")