from app.config.app import Config
from app.container import training_parameters, nn_resonator, training_data_generator, out_filepath, \
    file_storage, training_loss_series_provider, parameters_storage, trainer
from resonator_ml.core.use_cases.plot_training_data import PlotTrainingData
from resonator_ml.core.use_cases.plot_training_result import PlotTrainingResult
from resonator_ml.core.use_cases.plot_weights import PlotWeights
from resonator_ml.core.use_cases.sound_generation import GenerateSoundFile
from resonator_ml.core.use_cases.training import TrainLoopNetwork
from resonator_ml.machine_learning.loop_filter.neural_network import Trainer
import shutil
from utils.stdout_redirect import redirect_stdout_to_file

def build_train_loop_network_use_case(config: Config):


    training_params = training_parameters(config)

    resonator = nn_resonator(config, load_model_weights=False, initialize_resonator=False)
    storage = file_storage(config)

    # TODO clean up logging + versioning. Doesn't belong here at all...
    old_model_path = storage.model_file_path()
    storage.make_new_version_output_dir()
    if config.reuse_last_model_file:
        shutil.copyfile(old_model_path, storage.model_file_path())

    configure_stdout(config, 'train_loop_network')
    print (config)
    print(training_params)
    print (resonator.model)
    return TrainLoopNetwork(resonator.model, training_data_generator=training_data_generator(config), file_storage=storage,
                            trainer=trainer(config), params_storage=parameters_storage(config), app_config=config)

def build_generate_sound_file_use_case(config: Config):
    resonator = nn_resonator(config, load_model_weights=True, initialize_resonator=True)

    print(config)
    print(training_parameters(config))
    return GenerateSoundFile(resonator, file_storage=file_storage(config), samplerate=config.sample_rate, file_length=config.output_soundfile_length)

def build_plot_training_result_use_case(config: Config):

    return PlotTrainingResult(training_loss_series_provider(config))

def build_plot_weights_use_case(config: Config):

    return PlotWeights(nn_resonator(config, load_model_weights=False, initialize_resonator=False).model)

def build_plot_training_data_use_case(config: Config):

    return PlotTrainingData(training_data_generator(config))


def configure_stdout(config: Config, log_name: str):
    path = file_storage(config).model_file_path()

    redirect_stdout_to_file(path.parent.absolute().as_posix(), script_name=log_name)