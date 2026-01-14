from resonator_ml.machine_learning.loop_filter.training_data import TrainingDataGenerator, TrainingFileDescriptor
from resonator_ml.machine_learning.training import TrainingParameterFactory

if __name__ == "__main__":
    training_parameter_factory = TrainingParameterFactory()
    training_parameters = training_parameter_factory.create_parameters("v2.1")
    instrument = "Strat_E"
    file_descriptor = TrainingFileDescriptor(model_name=instrument, parameter_string='0')
    training_data_generator = TrainingDataGenerator(training_parameters, file_descriptor)
    dataset = training_data_generator.generate_training_dataset("v1",instrument=instrument)
    print(dataset)