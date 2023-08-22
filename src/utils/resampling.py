from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.utils import resample, shuffle
import pandas as pd


def undersample(df, target):
    df_majority = df[df[target] == 0]
    df_minority = df[df[target] == 1]

    df_majority_undersampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=123)
    df_undersampled = pd.concat([df_majority_undersampled, df_minority])

    return df_undersampled


def oversample(df, target):
    df_majority = df[df[target] == 0]
    df_minority = df[df[target] == 1]

    df_minority_oversampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=123)

    df_oversampled = pd.concat([df_majority, df_minority_oversampled])

    return df_oversampled


def smote(df, target):
    sm = SMOTE(random_state=42)
    X = df.drop(target, axis=1)
    y = df[target]
    X_res, y_res = sm.fit_resample(X, y)

    df_resampled = pd.DataFrame(X_res, columns=X.columns)
    df_resampled[target] = y_res
    df_resampled = shuffle(df_resampled, random_state=42)

    return df_resampled


def undersample_and_oversample(df, target):
    nm = NearMiss()
    sm = SMOTE(random_state=42)
    X = df.drop(target, axis=1)
    y = df[target]
    X_res, y_res = nm.fit_resample(X, y)
    X_res, y_res = sm.fit_resample(X_res, y_res)

    df_resampled = pd.DataFrame(X_res, columns=X.columns)
    df_resampled[target] = y_res
    df_resampled = shuffle(df_resampled, random_state=42)

    return df_resampled
