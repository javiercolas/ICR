import h2o
from h2o.automl import H2OAutoML
from pandas import DataFrame
from typing import List
from yaspin import yaspin
import sys
import io
from utils.logs import ColorLogger
import time


class H20Training:
    def __init__(self, df: DataFrame, X_columns: List[str], target: str, max_models: int = 15):
        self.logging = ColorLogger.get_logger(__name__)
        self.X = X_columns
        self.y = target
        self.max_models = max_models
        self.data = h2o.H2OFrame(df)
        self.logging.info(f"Initialized with X columns: {self.X}, target: {self.y}, max_models: {self.max_models}")

    @yaspin(text="Training...", spinner="dots")
    def train(self) -> H2OAutoML:
        try:
            self.data[self.y] = self.data[self.y].asfactor()
            aml = H2OAutoML(max_models=self.max_models, seed=1)
            original_stdout = sys.stdout
            sys.stdout = io.StringIO()

            start_time = time.time()
            aml.train(x=self.X, y=self.y, training_frame=self.data)
            end_time = time.time()
            self.logging.info(f"Training duration: {end_time - start_time}")

            sys.stdout = original_stdout
            return aml
        except Exception as error:
            self.logging.error(f"Error: {error}")
            h2o.shutdown()
            h2o.init()

