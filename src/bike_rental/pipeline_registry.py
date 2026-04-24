"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from .pipelines.feature_eng import *
from .pipelines.training import *
from .pipelines.inference import *


def register_pipelines() -> dict[str, Pipeline]:
    feature_eng_training = feat_eng_pipeline_training()
    feature_eng_inference = feat_eng_pipeline_inference()
    training_pipeline = create_training_pipeline()
    inference_pipeline = create_inference_pipeline()
    return {"__default__": feature_eng_training+ training_pipeline, 
     "training": feature_eng_training+ training_pipeline,
    "inference": feature_eng_inference + inference_pipeline}
