import numpy as np
import pytest

from sklearn.linear_model import LinearRegression

import shap.benchmark.models as models


# =========================
# Fixtures
# =========================

@pytest.fixture
def data():
    X = np.random.randn(50, 5)
    y = np.random.randn(50)
    return X, y


# =========================
# KerasWrap (core logic)
# =========================

def test_keras_wrap_basic():
    class DummyModel:
        def __init__(self):
            self.weights = [1]

        def get_weights(self):
            return self.weights

        def set_weights(self, w):
            self.weights = w

        def fit(self, X, y, epochs=1, verbose=0):
            return self

        def predict(self, X):
            return np.ones((X.shape[0], 1))

    model = DummyModel()
    wrap = models.KerasWrap(model, epochs=1, flatten_output=True)

    X = np.random.randn(10, 5)
    y = np.random.randn(10)

    wrap.fit(X, y)
    preds = wrap.predict(X)

    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]


# =========================
# sklearn model factories
# =========================

def test_corrgroups_models(data):
    X, y = data

    funcs = [
        models.corrgroups60__lasso,
        models.corrgroups60__ridge,
        models.corrgroups60__decision_tree,
        models.corrgroups60__random_forest,
    ]

    for fn in funcs:
        model = fn()
        model.fit(X, y)
        preds = model.predict(X)

        assert preds.shape[0] == X.shape[0]


# =========================
# independent linear models
# =========================

def test_independentlinear_models(data):
    X, y = data

    funcs = [
        models.independentlinear60__lasso,
        models.independentlinear60__ridge,
        models.independentlinear60__decision_tree,
        models.independentlinear60__random_forest,
    ]

    for fn in funcs:
        model = fn()
        model.fit(X, y)
        preds = model.predict(X)

        assert preds.shape[0] == X.shape[0]


# =========================
# cric models (predict override)
# =========================

def test_cric_models(data):
    X, y = data

    funcs = [
        models.cric__lasso,
        models.cric__ridge,
        models.cric__decision_tree,
        models.cric__random_forest,
    ]

    # convert y to classification labels
    y_class = (y > 0).astype(int)

    for fn in funcs:
        model = fn()
        model.fit(X, y_class)
        preds = model.predict(X)

        assert preds.shape[0] == X.shape[0]


# =========================
# optional: xgboost
# =========================

def test_xgboost_models_optional(data):
    pytest.importorskip("xgboost")

    X, y = data

    model = models.corrgroups60__gbm()
    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape[0] == X.shape[0]


# =========================
# optional: tensorflow
# =========================

def test_tensorflow_models_optional():
    tf = pytest.importorskip("tensorflow")

    model = models.corrgroups60__ffnn()

    # just check object creation
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")