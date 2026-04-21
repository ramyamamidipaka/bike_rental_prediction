"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from .pipelines.feature_eng import create_feature_eng_pipeline
from .pipelines.training import create_training_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    feature_eng_pipeline = create_feature_eng_pipeline()
    training_pipeline = create_training_pipeline()
    return {"__default__": feature_eng_pipeline+ training_pipeline, 
    "feature_eng": feature_eng_pipeline, "training": feature_eng_pipeline+ training_pipeline}
