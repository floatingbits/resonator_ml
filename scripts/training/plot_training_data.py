from resonator_ml.machine_learning.loop_filter.neural_network import prepare_dataloader
from resonator_ml.machine_learning.loop_filter.training_data import TrainingDataGenerator
from resonator_ml.machine_learning.view.audio import BatchFeatureViewer
import matplotlib.pyplot as plt

if __name__ == "__main__":
    instrument = "KS"
    resonator_type_name = "v1"
    model_suffix = "_9"
    # model_suffix = "_test_v09"



    training_data_generator = TrainingDataGenerator()
    dataset = training_data_generator.generate_training_dataset(network_type=resonator_type_name, instrument=instrument)
    dataloader = prepare_dataloader(dataset, batch_size=32)
    for inputs, target in dataloader:
        viewer = BatchFeatureViewer(inputs, target)
        break


    plt.show()
