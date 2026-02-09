from app.bootstrap import build_train_loop_network_use_case
from app.config.app import Config as AppConfig


def run(config: AppConfig):
    train = build_train_loop_network_use_case(config)

    train.execute()
