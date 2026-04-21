from kedro.pipeline import Pipeline, node
from .nodes import *
from typing import Dict, Any, Tuple
import pandas as pd

def create_training_pipeline() -> Pipeline:
    return Pipeline([
        node(
            func=make_target,
            inputs=["features", "params:training.target_params"],
            outputs="data_with_target"
        ),
        node(
            func=split_data,
            inputs=["data_with_target", "params:training"],
            outputs=["x_train", "x_test", "y_train", "y_test"],
        ),
        node(
            func=tune_hyperparameters,
            inputs=["x_train", "y_train", "x_test", "y_test", "params:training"],
            outputs="best_params",
        ),
        node(
            func=train_model,
            inputs=["x_train", "y_train", "best_params"],
            outputs=["trained_model", "model_type"],
        ),
        node(
            func=predict,
            inputs=["trained_model", "x_test"],
            outputs="predictions",
        ),
        node(
            func=compute_metrics,
            inputs=["y_test", "predictions"],
            outputs="metrics",
        ),
        node(
            func=save_model,
            inputs=["trained_model", "model_type", "best_params", "params:model_storage"],
            outputs=None,
        ),
    ])