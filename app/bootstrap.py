from app.container import training_parameters, nn_resonator, model_file_name, training_data_generator, out_filepath, app_config
from resonator_ml.core.use_cases.sound_generation import GenerateSoundFile
from resonator_ml.core.use_cases.training import TrainLoopNetwork
from resonator_ml.machine_learning.loop_filter.neural_network import Trainer



def build_train_loop_network_use_case():


    training_params = training_parameters()

    resonator = nn_resonator(load_model_weights=False, initialize_resonator=False)

    trainer = Trainer(training_params)

    return TrainLoopNetwork(resonator.model, training_data_generator=training_data_generator(), model_file_name=model_file_name(), trainer=trainer)

def build_generate_sound_file_use_case():
    resonator = nn_resonator(load_model_weights=True, initialize_resonator=True)
    config = app_config()
    return GenerateSoundFile(resonator, out_filepath=out_filepath(), samplerate=config.sample_rate)
