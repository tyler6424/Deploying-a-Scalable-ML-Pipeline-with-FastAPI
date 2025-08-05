import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)
# TODO: load the cencus.csv data
root_dir = Path(__file__).resolve().parent
data_path = root_dir / "data" / "census.csv"
print(data_path)
df = pd.read_csv(data_path).replace("?", "Unknown")

# TODO: split the provided data to have a train dataset and a test dataset
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train_df, test_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["salary"]
)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
label="salary"

# TODO: use the process_data function provided to process the data.
X_train, y_train, encoder, lb = process_data(
    train_df,
    categorical_features=cat_features,
    label=label,
    training=True
)

X_test, y_test, _, _ = process_data(
    test_df,
    categorical_features=cat_features,
    label=label,
    training=False,
    encoder=encoder,
    lb=lb,
)

# TODO: use the train_model function to train the model on the training dataset
model = train_model(X_train, y_train)

# save the model and the encoder
model_dir = project_path / "model"
save_model(model, encoder, lb, model_dir=model_dir)

# load the model
model, encoder, lb = load_model(model_dir=model_dir)

# TODO: use the inference function to run the model inferences on the test dataset.
preds = inference(model, X_test)

# Calculate and print the metrics
prec, rec, f1 = compute_model_metrics(y_test, preds)
print(f"Overall  P:{prec:.3f}  R:{rec:.3f}  F1:{f1:.3f}")

# TODO: compute the performance on model slices using the performance_on_categorical_slice function
# iterate through the categorical features

with open("slice_output.txt", "a") as f:
    for col in cat_features:
        for slicevalue in sorted(test_df[col].unique()):
            count = test_df[test_df[col] == slicevalue].shape[0]

            p, r, fb, _ = performance_on_categorical_slice(
                test_df,
                col,
                slicevalue,
                cat_features,
                label,
                encoder, lb, model
            )

            # write two lines exactly like the starter template
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)