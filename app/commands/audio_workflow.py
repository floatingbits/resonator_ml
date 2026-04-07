from app.config.app import Config as AppConfig
from app.orchestration.simple import DefaultAudioOrchestrator


def run(config: AppConfig):
    DefaultAudioOrchestrator().run(config)

