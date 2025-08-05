import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

from ml.data import process_data
from ml.model import (
    train_model,
    inference,
    performance_on_categorical_slice,
)

CATS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]
LABEL = "salary"

# load just 500 rows to keep tests quick
DF = (
    pd.read_csv("data/census.csv")
    .replace("?", "Unknown")
    .sample(n=500, random_state=1)
)


# TODO: implement the first test. Change the function name and input as needed
def test_train_model_type():
    X, y, _, _ = process_data(DF, CATS, label=LABEL, training=True)
    model = train_model(X, y)
    assert isinstance(model, ClassifierMixin), "train_model should return an sklearn classifier"


# TODO: implement the second test. Change the function name and input as needed
def test_inference_shape():
    X, y, _, _ = process_data(DF, CATS, label=LABEL, training=True)
    model = train_model(X, y)
    preds = inference(model, X[:10])
    assert preds.shape == (10,), "inference should return 1 prediction per row"


# TODO: implement the third test. Change the function name and input as needed
def test_slice_metrics():
    X, y, enc, lb = process_data(DF, CATS, label=LABEL, training=True)
    model = train_model(X, y)
    p, r, f1, cnt = performance_on_categorical_slice(
        DF, "sex", "Male", CATS, LABEL, enc, lb, model
    )
    # precision / recall / f1 are between 0 and 1, count is non-negative
    assert 0.0 <= p <= 1.0 and 0.0 <= r <= 1.0 and 0.0 <= f1 <= 1.0
    assert cnt >= 0
