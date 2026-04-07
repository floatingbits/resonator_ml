from dataclasses import dataclass
from typing import Callable

from app.config.app import Config


@dataclass
class ExperimentDefinition:
    runs_per_config: int
    configs: list
    config_extractor: Callable[[Config], dict]
