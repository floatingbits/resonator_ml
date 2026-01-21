from app.container import training_parameters, nn_resonator, training_data_generator, out_filepath, \
    app_config, file_storage, training_loss_series_provider, parameters_storage
from resonator_ml.core.use_cases.plot_training_result import PlotTrainingResult
from resonator_ml.core.use_cases.sound_generation import GenerateSoundFile
from resonator_ml.core.use_cases.training import TrainLoopNetwork
from resonator_ml.machine_learning.loop_filter.neural_network import Trainer

from utils.stdout_redirect import redirect_stdout_to_file

def build_train_loop_network_use_case():


    training_params = training_parameters()

    resonator = nn_resonator(load_model_weights=False, initialize_resonator=False)
    trainer = Trainer(training_params)
    storage = file_storage()
    # TODO clean up logging + versioning
    storage.make_new_version_output_dir()
    configure_stdout('train_loop_network')
    print (app_config())
    print(training_params)
    print (resonator.model)
    return TrainLoopNetwork(resonator.model, training_data_generator=training_data_generator(), file_storage=storage,
                            trainer=trainer, params_storage=parameters_storage(), app_config=app_config())

def build_generate_sound_file_use_case():
    resonator = nn_resonator(load_model_weights=True, initialize_resonator=True)
    config = app_config()
    print(app_config())
    print(training_parameters())
    return GenerateSoundFile(resonator, file_storage=file_storage(), samplerate=config.sample_rate, file_length=config.output_soundfile_length)

def build_plot_training_result_use_case():

    return PlotTrainingResult(training_loss_series_provider())


def configure_stdout(log_name: str):
    path = file_storage().model_file_path()

    redirect_stdout_to_file(path.parent.absolute().as_posix(), script_name=log_name)