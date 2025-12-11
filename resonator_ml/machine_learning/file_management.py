MODELS_DIRECTORY = "models"
LOOP_FILTER_NN_DIRECTORY = "loop_filter_nn"

def create_loop_filter_model_file_name(version:str):
    return ('{models_dir}/{loop_filter_dir}/{version}.pt'
            .format(models_dir=MODELS_DIRECTORY, loop_filter_dir= LOOP_FILTER_NN_DIRECTORY, version=version))