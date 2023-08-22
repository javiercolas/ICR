import pandas as pd
from pandas import DataFrame


class Prepare:
    def __init__(self, source_data_path: str):
        self.source_data_path = source_data_path

    def prepare(self) -> DataFrame:
        return pd.read_csv(self.source_data_path)
