from resonator_ml.ports.evaluation_data_provider import EvaluationDataProvider, EvaluationData
from pathlib import Path
import json
class ExperimentDataProvider(EvaluationDataProvider):
    def __init__(self, search_path: Path):
        self.search_path = search_path


    def files(self):
        return sorted(list(self.search_path.rglob("results.json")), key=lambda p: int(p.parent.name))
    def data_at(self, index: int) -> dict:
        filepath = self.files()[index]
        with open(filepath, 'r', encoding="utf-8") as file:
            return json.load(file)

    def num_results(self) -> int:
        return len(self.files())
    def provide_data(self) -> EvaluationData:
        base = self.data_at(0)
        data = {}
        input_fields = list(base['config'].keys())
        result_fields = list(base['results'].keys())
        group_id_field = "config_id"
        in_group_index_field = "config_run_id"
        fields = input_fields + result_fields + [group_id_field, in_group_index_field]
        for index in range(self.num_results()):
            file_data = self.data_at(index)
            for field in fields:
                if field not in data.keys():
                    data[field] = []
                if field in input_fields:
                    val = file_data['config'][field]
                elif field in result_fields:
                    val = file_data['results'][field]
                else:
                    val = file_data[field]
                data[field].append(val)

        return EvaluationData(
            input_fields=input_fields,
            result_fields=result_fields,
            group_id_field="config_id",
            in_group_index_field="config_run_id",
            data=data
        )

