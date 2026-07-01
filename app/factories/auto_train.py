from app.config.app import Config
from app.factories.storage import file_storage
from app.orchestration.auto_training.auto_trainer import AutoTrainer


def auto_trainer(config: Config):
    return AutoTrainer(config, file_storage=file_storage(config))