import dataclasses
from typing import Protocol

@dataclasses.dataclass
class EvaluationData:
    input_fields: list[str]
    result_fields: list[str]
    group_id_field: str
    in_group_index_field: str
    data: dict[str, list]

class EvaluationDataProvider(Protocol):
    def provide_data(self) -> EvaluationData:...