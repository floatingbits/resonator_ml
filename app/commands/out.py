from app.bootstrap import build_generate_sound_file_use_case
from app.config.app import Config as AppConfig


def run(config: AppConfig):
    use_case = build_generate_sound_file_use_case(config)
    use_case.execute()
