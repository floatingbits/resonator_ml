from app.config.app import Config
from app.factories.experiment_result_data import experiment_result_data_provider
from app.factories.metrics import audio_perceptual_metrics
from app.factories.training_data import init_sound_file, training_data_generator
from app.factories.storage import file_storage, parameters_storage, results_storage
from app.factories.resonator import nn_resonator
from app.factories.training import trainer, training_parameters, \
    training_loss_series_provider, build_predictor, loss_module, TrainingRunContext, dynamic_loss_scheduler
from resonator_ml.core.use_cases.compare_models import CompareModels
from resonator_ml.core.use_cases.compute_metrics import ComputeMetrics
from resonator_ml.core.use_cases.plot_spectrum_comparison import PlotSpectrumComparison
from resonator_ml.core.use_cases.plot_training_data import PlotTrainingData
from resonator_ml.core.use_cases.plot_training_result import PlotTrainingResult
from resonator_ml.core.use_cases.plot_weights import PlotWeights
from resonator_ml.core.use_cases.show_result_metrics import ShowResultMetrics
from resonator_ml.core.use_cases.sound_generation import GenerateSoundFile
from resonator_ml.core.use_cases.training import TrainLoopNetwork

from pathlib import Path
import shutil
import torch

from resonator_ml.machine_learning.loop_filter.neural_network import NNResonatorInitializer
from resonator_ml.machine_learning.training.trainer import SimpleOptimizationPolicy, SimpleModelPersistenceManager
from utils.stdout_redirect import redirect_stdout_to_file


def build_train_loop_network_use_case(config: Config):


    training_params = training_parameters(config)

    storage = file_storage(config)


    # TODO clean up logging + versioning. Doesn't belong here at all...
    copy_from_old = config.reuse_last_model_file or config.src_model_file_path is not None
    try:
        old_model_path = storage.src_model_file_path()
    except FileNotFoundError as e:
        if copy_from_old:
            raise FileNotFoundError('Cannot reuse last model file because it cannot be found.')
    storage.make_new_version_output_dir()
    if copy_from_old:
        shutil.copyfile(old_model_path, storage.model_file_path())
        # copy also a backup to be able to reuse the last backup
        shutil.copyfile(old_model_path, storage.model_file_path().parent / (storage.model_file_path().name + '.bak'))

    # ATM only initialize after new version output dir is stored
    training_run_context = build_training_run_context(config)
    configure_stdout(config, 'train_loop_network')
    print (config)
    print(training_params)
    print (training_run_context.model)
    return TrainLoopNetwork(training_run_context.model, file_storage=storage,
                            trainer=trainer(training_run_context), params_storage=parameters_storage(config), app_config=config)

def build_generate_sound_file_use_case(config: Config):
    resonator = nn_resonator(config)


    model = resonator.model
    model.load_state_dict(torch.load(file_storage(config).model_file_path(), weights_only=True))
    model.eval()


    filepath = init_sound_file(config)
    initializer = NNResonatorInitializer()
    initializer.initialize(resonator, filepath)
    print(config)
    print(training_parameters(config))
    return GenerateSoundFile(resonator, file_storage=file_storage(config), samplerate=config.sample_rate, file_length=config.output_soundfile_length)

def build_plot_training_result_use_case(config: Config):

    return PlotTrainingResult(training_loss_series_provider(config))

def build_plot_weights_use_case(config: Config):

    return PlotWeights(nn_resonator(config).model)

def build_show_result_metrics_use_case(config: Config):
    return ShowResultMetrics(experiment_result_data_provider(config), training_loss_series_provider(config))

def build_plot_training_data_use_case(config: Config):

    return PlotTrainingData(training_data_generator(config))

def build_plot_spectrum_comparison(config: Config):
    output_file = file_storage(config).sound_output_path().as_posix()
    init_file = init_sound_file(config)
    return PlotSpectrumComparison({'Output': output_file, 'init':  init_file})

def build_compute_metrics_use_case(config: Config):
    output_file = file_storage(config).sound_output_path().as_posix()
    init_file = init_sound_file(config)
    metrics = audio_perceptual_metrics()
    return ComputeMetrics(init_file,output_file, metrics, result_storage=results_storage(config))

def build_compare_models_use_case(config: Config):
    paths = {
        'low_loss': Path('./data/results/resonator/workspace/experiments/synthetic_training/Strat_E/2/30/model.pt'),
        'high_loss': Path('./data/results/resonator/workspace/experiments/synthetic_training/Strat_E/2/29/model.pt')
    }
    predictors = {}
    for path_key in paths:
        path = paths[path_key]
        resonator = nn_resonator(config)
        model = resonator.model
        if path.exists():
            model.load_state_dict(torch.load(path))
        else:
            raise FileNotFoundError(path)
        predictors[path_key] = build_predictor(model)
    dataloader,test_data = training_data_generator(config).generate_training_dataloader()
    return CompareModels(predictors,dataloader,loss_module=loss_module(config))


def configure_stdout(config: Config, log_name: str):
    try:
        path = file_storage(config).model_file_path()
        redirect_stdout_to_file(path.parent.absolute().as_posix(), script_name=log_name)
    except FileNotFoundError as e:
        # TODO: This stdout redirect is a bit messy and it is unclear, when it is really to be called
        # or whose responsibility it is. This catch block is only a workaround when this function is called
        # before the output dir exists (experiment runs)
        # replace with proper logging so we won't need any messy redirect in advance
        pass


def build_training_run_context(config: Config):
    resonator = nn_resonator(config)
    model = resonator.model
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training_parameters.learning_rate)
    policy = SimpleOptimizationPolicy(optimizer)
    predictor = build_predictor(model)
    persistence_manager = SimpleModelPersistenceManager(file_storage(config).model_file_path().as_posix(), model)
    dataloader, test_data = training_data_generator(config).generate_training_dataloader()
    loss_mod = loss_module(config)
    loss_scheduler = dynamic_loss_scheduler(config, loss_mod)
    return TrainingRunContext(model, optimizer, policy, predictor,
                              loss_mod, persistence_manager, config.training_parameters.epochs, dataloader, test_data, loss_scheduler)
