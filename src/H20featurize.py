from utils.transformations import *
import pandas as pd


class H20Featurize:
    def __init__(self, df, transformation_columns=None, transformation_technique=None):
        self.df = df
        self.transformation_columns = transformation_columns
        self.transformation_technique = transformation_technique

    def featurize(self):
        self._convert_and_rearrange_categorical_features('EJ', 'Class')
        self._impute_missing_values()
        self._balance_classes()
        self._transform_variables()
        return self.df

    def _balance_classes(self, apply=False):
        if not apply:
            pass
        else:
            pass
            # apply balance class techniques

    def _impute_missing_values(self, apply=False):
        if not apply:
            pass
        else:
            pass
            # apply missing values imputation techniques

    def _transform_variables(self, apply=False):
        if not apply:
            pass
        else:
            pass

    def _convert_and_rearrange_categorical_features(self, categorical_feature, target):
        self.df = pd.get_dummies(self.df, columns=[categorical_feature], drop_first=True)
        dummy_column = f"{categorical_feature}_B"
        if dummy_column in self.df.columns:
            self.df.rename(columns={dummy_column: categorical_feature}, inplace=True)

        cols = [col for col in self.df if col != target] + [target]
        self.df = self.df[cols]

