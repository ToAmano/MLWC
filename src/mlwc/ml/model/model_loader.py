import torch

from mlwc.ml.model.mlmodel_basic_descs import ModelAHandler

MODEL_REGISTRY = {
    "NET_withoutBN_descs": ModelAHandler,
}


def load_model_handler(path: str):
    checkpoint = torch.load(path, map_location="cpu")
    model_type = checkpoint.get("model_type")

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")

    model_cls = MODEL_REGISTRY[model_type]
    return model_cls.load_from_file(path)
