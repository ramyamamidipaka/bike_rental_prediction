from kedro.pipeline import Pipeline, node

from .nodes import *

def create_inference_pipeline(
    
):
    """Create the inference pipeline."""
    return Pipeline([
        node(
            func=load_model,
            inputs=["model_type", "params:model_storage"],
            outputs="model",
            name="load_model",
        ),
        node(
            func=predict,
            inputs=["model", "features"],
            outputs="predictions",
        ),
        node(
            func=join_timestamps,
            inputs=["predictions", "timestamps"],
            outputs="predictions_with_timestamps",
        ),
    ])