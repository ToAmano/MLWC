import inspect
from typing import Any

from .registry import TRANSFORM_REGISTRY


def get_transform(name: str, **kwargs) -> Any:
    """
    Gets a transform class from the registry by name and instantiates it.

    This factory is useful for config-driven workflows (e.g., training)
    where parameters are explicitly provided in a configuration file.

    Args:
        name (str): The registered name of the transform.
        **kwargs: Keyword arguments to be passed to the transform's __init__ method.

    Returns:
        An instance of the corresponding transform class.
    """
    if name not in TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown transform name: '{name}' is not registered.")

    transform_class = TRANSFORM_REGISTRY[name]

    # Instantiate the class with the provided keyword arguments
    return transform_class(**kwargs)


def build_transform_from_model(model: Any) -> Any:
    """
    Builds the appropriate transform for a given model object automatically.

    This intelligent factory inspects the model for metadata (`transform_name`
    and other required parameters) to build its corresponding transform.
    This is ideal for inference/prediction workflows to avoid needing a config file.

    Args:
        model: The trained model object. Must have a `transform_name` attribute,
               and attributes matching the names of its transform's __init__ args.

    Returns:
        A fully configured instance of the transform required by the model.
    """
    # Convention 1: Model must store the name of the transform it needs.
    # transform_name = getattr(model, "transform_name", None)
    transform_name = getattr(model, "modeltype", None)
    if transform_name is None:
        raise AttributeError(
            "The provided model object does not have a 'transform_name' attribute."
        )

    if transform_name not in TRANSFORM_REGISTRY:
        raise ValueError(
            f"Transform '{transform_name}' found on model is not registered."
        )

    transform_class = TRANSFORM_REGISTRY[transform_name]

    # Inspect the transform's constructor to find required arguments.
    init_signature = inspect.signature(transform_class.__init__)
    init_arg_names = [
        p.name for p in init_signature.parameters.values() if p.name != "self"
    ]

    # Convention 2: Model must have attributes matching the transform's __init__ arguments.
    transform_kwargs = {}
    for arg_name in init_arg_names:
        if hasattr(model, arg_name):
            transform_kwargs[arg_name] = getattr(model, arg_name)
        else:
            # __init__が必要とする引数がモデルに見つからなかった場合
            raise AttributeError(
                f"Model is missing attribute '{arg_name}' which is required by "
                f"'\{transform_class.__name__}.__init__'."
            )

    # Instantiate the transform with the arguments gathered from the model.
    return transform_class(**transform_kwargs)
