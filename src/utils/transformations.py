from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, KBinsDiscretizer, QuantileTransformer
import numpy as np


def standardization(df, column_name):
    scaler = StandardScaler()
    return scaler.fit_transform(df[[column_name]])


def min_max_scaling(df, column_name):
    scaler = MinMaxScaler()
    return scaler.fit_transform(df[[column_name]])


def clipping(df, column_name, lower_percentile, upper_percentile):
    lower = np.percentile(df[column_name], lower_percentile)
    upper = np.percentile(df[column_name], upper_percentile)
    return np.clip(df[column_name], lower, upper)


def binning(df, column_name, n_bins):
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    return discretizer.fit_transform(df[[column_name]])


def rank(df, column_name):
    return df[column_name].rank()


def logarithmic_transformation(df, column_name):
    return np.log1p(df[column_name])  # log(1+x)


def box_cox_transformation(df, column_name):
    pt = PowerTransformer(method='box-cox')
    return pt.fit_transform(df[[column_name]])


def yeo_johnson_transformation(df, column_name):
    pt = PowerTransformer(method='yeo-johnson')
    df[column_name] = pt.fit_transform(df[[column_name]])
    return df


def rank_gauss(df, column_name):
    transformer = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=0)
    df[column_name] = transformer.fit_transform(df[[column_name]])
    return df
