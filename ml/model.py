
import joblib
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from ml.data import process_data

# Optional: implement hyperparameter tuning.
def train_model(X_train: np.ndarray, y_train: np.ndarray):
    """
    Trains a RandomForest classifier and returns it.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model

def inference(model, X: np.ndarray) -> np.ndarray:
    """Run model inference."""
    return model.predict(X)


def compute_model_metrics(y: np.ndarray, preds: np.ndarray):
    """
    Returns precision, recall and F-beta (F1).
    """
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    return precision, recall, fbeta


def save_model(model, encoder, lb, model_dir: str = "model"):
    """
    Saves the model and preprocessing artefacts to `model_dir/`.
    """
    Path(model_dir).mkdir(exist_ok=True)
    joblib.dump(model, Path(model_dir) / "model.pkl")
    joblib.dump(encoder, Path(model_dir) / "encoder.pkl")
    joblib.dump(lb, Path(model_dir) / "lb.pkl")


def load_model(model_dir: str = "model"):
    """
    Loads the model and preprocessing artefacts from `model_dir/`.

    Returns
    -------
    model, encoder, lb
    """
    model = joblib.load(Path(model_dir) / "model.pkl")
    encoder = joblib.load(Path(model_dir) / "encoder.pkl")
    lb = joblib.load(Path(model_dir) / "lb.pkl")
    return model, encoder, lb


def performance_on_categorical_slice(
    data,
    column_name: str,
    slice_value,
    categorical_features: list,
    label: str,
    encoder,
    lb,
    model):
    """
    Computes precision, recall, F1 on the rows where
    `data[column_name] == slice_value`.
    """
    slice_df = data[data[column_name] == slice_value]
    if slice_df.empty:
        return 0.0, 0.0, 0.0, 0  # nothing to evaluate

    X_slice, y_slice, _, _ = process_data(
        slice_df,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    preds = inference(model, X_slice)
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta, len(slice_df)
