from resonator_ml.machine_learning.loop_filter.training_data import TrainingDataGenerator, prepare_dataloader, \
    TrainingFileDescriptor
from resonator_ml.machine_learning.training import TrainingParameters
from resonator_ml.machine_learning.view.audio import BatchFeatureViewer
import matplotlib.pyplot as plt

if __name__ == "__main__":
    instrument = "KS_E"
    resonator_type_name = "v1"
    model_suffix = "_9"
    # model_suffix = "_test_v09"

    training_parameters = TrainingParameters(batch_size=32, epochs=10, learning_rate=1e-3)
    file_descriptor = TrainingFileDescriptor(model_name=instrument, parameter_string='0')
    training_data_generator = TrainingDataGenerator(training_parameters, file_descriptor)
    dataloader = training_data_generator.generate_training_dataloader(network_type=resonator_type_name, instrument=instrument)

    for inputs, target in dataloader:
        viewer = BatchFeatureViewer(inputs, target)
        break


    plt.show()
