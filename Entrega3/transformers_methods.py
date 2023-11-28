import numpy as np
from sklearn.preprocessing import FunctionTransformer


def sin_transform(x, period):
    return np.sin(x / period * 2 * np.pi)


def cos_transform(x, period):
    return np.cos(x / period * 2 * np.pi)


def sin_transformer(period):
    return FunctionTransformer(sin_transform, feature_names_out="one-to-one", kw_args=dict(period=period))


def cos_transformer(period):
    return FunctionTransformer(cos_transform, feature_names_out="one-to-one", kw_args=dict(period=period))
