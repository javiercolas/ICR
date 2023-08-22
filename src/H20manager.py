from H20featurize import H20Featurize
from prepare import Prepare
from H20train import H20Training
from H20evaluate import H20Evaluate
from H20display import H20Display
from mrmr import mrmr_classif
from itertools import combinations
import os
import tqdm
import json
import h2o
from typing import List
import yaml
from utils.logs import ColorLogger


class Manager:
    def __init__(self, src_csv_path='data/src/train.csv'):
        self.logging = ColorLogger.get_logger(__name__)
        self._restart_h2o()
        self.H20Evaluate = None
        self.H20Training = None
        self.H20featurize = None
        self.prepare = None
        self.mrmr_priority_features = None
        self.dropped_features = ('Id', 'Class')
        self.target = 'Class'
        self.src_csv_path = src_csv_path
        h2o.init()

    def _restart_h2o(self):
        try:
            h2o.shutdown(prompt=False)
        except Exception as error:
            self.logging.error('Failed to shut down an H2O cluster:', error)
            try:
                h2o.init()
                self.logging.info('Successfully restarted H2O cluster.')
            except Exception as error:
                self.logging.error('Failed to restart H2O cluster:', error)
                raise

    def execute_pipeline(self):
        self.logging.info('Executing pipeline...')
        self.prepare = Prepare(source_data_path=self.src_csv_path).prepare()
        self.H20featurize = H20Featurize(df=self.prepare).featurize()
        self._set_mrmr_priority_features()
        self._recursive_pipeline()

    def _pipeline(self, X_columns):
        self.H20Training = H20Training(df=self.H20featurize, X_columns=X_columns, target='Class', max_models=10).train()
        self.H20Evaluate = H20Evaluate(auto_ml=self.H20Training, X_columns=X_columns, target='Class').response()
        H20Display(evaluate_info=self.H20Evaluate, X_columns=X_columns).display()

    def _set_mrmr_priority_features(self):
        X = self.H20featurize.drop(list(self.dropped_features), axis=1)
        y = self.H20featurize[self.target]
        self.mrmr_priority_features = mrmr_classif(X=X, y=y, K=len(X.columns), return_scores=False)

    def _check_unused_features(self, candidates: List, results_path: str = 'data/results/results.json') -> List:
        unused_features = []
        used_features = []
        candidates_list = [list(candidate) for candidate in candidates]

        if os.path.isfile(results_path):
            with open(results_path, 'r') as json_file:
                results_data = json.load(json_file)

            for feature_key in results_data.keys():
                used_feature_list = feature_key.split('_')[1:]
                used_features.append(used_feature_list)
                self.logging.info('Model trained with features: %s', used_feature_list)

            self.logging.info('Total number of models trained: %d', len(used_features))
            for candidate in candidates_list:
                if set(candidate) not in [set(x) for x in used_features]:
                    self.logging.debug('Unused feature: %s', candidate)
                    unused_features.append(candidate)
        else:
            self.logging.error('The file does not exist: %s', results_path)
            unused_features = candidates_list

        self.logging.info('Models remaining: %d', len(unused_features))
        return unused_features

    def _recursive_pipeline(self):
        self.logging.info('Starting recursive pipeline...')
        for feature_combinations in tqdm.tqdm(self._check_unused_features(candidates=self._mandatory_features_combinations() + self._custom_priority_features() + self._features_combination())):
            try:
                self._pipeline(X_columns=list(feature_combinations))
            except Exception as error:
                self.logging.error('Error in recursive pipeline: %s', error)

    def _features_combination(self, max_features_combination: int = 13) -> List:
        self.logging.info('Generating feature combinations...')
        variables = self.mrmr_priority_features[:max_features_combination]
        possible_features = []
        for r in range(1, len(variables) + 1):
            for combination in combinations(variables, r):
                possible_features.append(combination)
        self.logging.info('Total number of feature combinations: %d', len(possible_features))
        return possible_features

    def _custom_priority_features(self, combination_file_name: str = 'utils/custom_combinations.yaml') -> List:

        if os.path.isfile(combination_file_name):
            self.logging.info('Setting custom combination...')
            with open(combination_file_name, 'r') as file:
                combinations_dict = yaml.safe_load(file).values()
            return combinations_dict
        else:
            self.logging.info('No custom combination required')
            return []

    def _mandatory_features_combinations(self) -> List:
        return [self.mrmr_priority_features[:i + 2] for i in range(len(self.mrmr_priority_features))]


if __name__ == "__main__":
    Manager().execute_pipeline()
