# src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_raw(path):
    df = pd.read_csv(path)
    # normalize column names maybe
    return df

class TimeFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X['creation_time'] = pd.to_datetime(X['creation_time'], utc=True)
        X['end_time'] = pd.to_datetime(X['end_time'], utc=True)
        X['session_duration'] = (X['end_time'] - X['creation_time']).dt.total_seconds().fillna(0)
        X['bytes_sum'] = (X['bytes_in'].fillna(0) + X['bytes_out'].fillna(0))
        X['bytes_ratio'] = X['bytes_in'] / (X['bytes_out'] + 1)
        return X

def build_pipeline():
    numeric = ['bytes_in', 'bytes_out', 'session_duration', 'bytes_sum', 'bytes_ratio']
    cat = ['protocol', 'src_ip_country_code']

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    from sklearn.compose import ColumnTransformer
    preproc = ColumnTransformer([
        ('num', numeric_pipeline, numeric),
        ('cat', cat_pipeline, cat)
    ])
    full = Pipeline([('timefeat', TimeFeatures()), ('preproc', preproc)])
    return full
