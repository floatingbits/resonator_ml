from resonator_ml.machine_learning.loop_filter.training_data import TrainingDataGenerator

if __name__ == "__main__":
    training_data_generator = TrainingDataGenerator()
    dataset = training_data_generator.generate_training_dataset()
    print(dataset)