import json
import pandas as pd


class ExploreResults:
    def __init__(self, json_file_path: str = 'data/results/results.json', output_path: str = 'data/results/metrics.csv'):
        self.json_file_path = json_file_path
        self.output_path = output_path

    def explore_results(self):
        self._result_metrics_to_csv()

    def _result_metrics_to_csv(self):
        with open(self.json_file_path, 'r') as file:
            data = json.load(file)

        pd.DataFrame({
            'columns_used': [model.split('_')[1:] for model in data.keys()],
            'logloss': self._own_compressed_list(data, 'logloss'),
            'precision': self._own_compressed_list(data, 'precision'),
            'recall': self._own_compressed_list(data, 'recall'),
            'F1': self._own_compressed_list(data, 'F1'),
            'accuracy': self._own_compressed_list(data, 'accuracy')
        }).to_csv(self.output_path)

    @staticmethod
    def _own_compressed_list(data, metric):
        all_metrics = []
        for model in data.values():
            model_metrics = []
            for i in model:
                model_metrics.append(i['metrics'][metric])
            all_metrics.append(model_metrics)
        return all_metrics


if __name__ == "__main__":
    ExploreResults().explore_results()



