from kedro.pipeline import Pipeline, node
from typing import Dict,Any
import pandas as pd
from .nodes import rename_columns,get_features

def create_feature_eng_pipeline() -> Pipeline:
    return Pipeline(
        [node(
            func=rename_columns,
            inputs=["train_data", "params:feature_engineering.rename_columns"],
            outputs="raw_data_renamed"
        ),
        node(
            func=get_features,
            inputs=["raw_data_renamed", "params:feature_engineering.lag_params"],
            outputs=["features", "timestamps"],
        )]
    )