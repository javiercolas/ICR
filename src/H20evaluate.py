import h2o
from h2o.automl import H2OAutoML
from typing import List
from utils.logs import ColorLogger


class H20Evaluate:
    """Class for evaluating H20 models."""

    def __init__(self, auto_ml: H2OAutoML, X_columns: List[str], target: str):
        self.logging = ColorLogger.get_logger(__name__)
        self.best_models = None
        self.auto_ml = auto_ml
        self.X_columns = X_columns
        self.target = target

    def get_and_store_best_models(self, sort_column: str = 'logloss'):
        """Get and store the best models based on a specified sort column."""
        lb = self.auto_ml.leaderboard.sort(sort_column)
        self.best_models = lb.head(rows=lb.nrows)

    def get_top_models_details(self, n_models: int = 5):
        """Get details of top models."""
        models_details = []
        for i in range(min(n_models, self.best_models.nrows)):
            model_id = self.best_models[i, 'model_id']
            models_details.append(self.get_model_and_base_model_details(model_id))
        return models_details

    def get_model_and_base_model_details(self, model):
        """Get details of a model and its base models if any."""
        model_performance, model_parameters = self.get_model_details(model)
        base_models_performance = []
        base_models_parameters = []
        if model_parameters['model_id'].startswith('StackedEnsemble'):
            base_models = self.get_base_models(model_parameters['base_models'])
            base_models_performance = [self.classification_metrics(perf) for perf, _ in base_models]
            base_models_parameters = [params for _, params in base_models]
        return model_performance, model_parameters, base_models_performance, base_models_parameters

    def get_base_models(self, base_models):
        """Get details of base models."""
        return [self.get_model_details(base_model['name']) for base_model in base_models]

    @staticmethod
    def get_model_details(model_name: str):
        """Get performance and parameters of a model."""
        model = h2o.get_model(model_name)
        return model.model_performance(), model.actual_params

    @staticmethod
    def classification_metrics(performance):
        """Get classification metrics of a model."""
        return {
            'accuracy': performance.accuracy(),
            'logloss': performance.logloss(),
            'F1': performance.F1(),
            'precision': performance.precision(),
            'recall': performance.recall()
        }

    def response(self):
        """Generate a response with top model details."""
        self.logging.info("Starting model evaluation...")
        self.get_and_store_best_models()
        top_models_details = self.get_top_models_details()
        self.logging.info("Model evaluation completed.")

        # Fetching best model's logloss
        best_model_logloss = self.best_models[0, 'logloss']
        self.logging.info(f"Best model's logloss: {best_model_logloss}")

        return [self.generate_response_dict(performance, parameters, base_models_performance, base_models_parameters)
                for performance, parameters, base_models_performance, base_models_parameters in top_models_details]

    def generate_response_dict(self, performance, parameters, base_models_performance, base_models_parameters):
        """Generate a dictionary response for a model."""
        return {
            "model": parameters['model_id'],
            "columns": {'X_columns': self.X_columns, 'target': self.target},
            "metrics": self.classification_metrics(performance),
            "base_models_metrics": self.fix_metrics_format(base_models_performance, base_models_parameters),
            "parameters": parameters,
            "base_models_parameters": self.fix_parameters_format(base_models_parameters)
        }

    @staticmethod
    def fix_metrics_format(base_models_performance, base_models_parameters):
        """Fix format of base model metrics."""
        return {param['model_id']: metric for metric, param in zip(base_models_performance, base_models_parameters)}

    @staticmethod
    def fix_parameters_format(base_models_parameters):
        """Fix format of base model parameters."""
        return {param['model_id']: param for param in base_models_parameters}
