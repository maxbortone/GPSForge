import GPSKet as qk
from typing import Union, Tuple
from netket.utils.types import Array
from ml_collections import ConfigDict


def get_dataset(name: str, config: ConfigDict) -> Union[Tuple[Array, Array], Array]:
    """
    Return a dataset for a system

    Args:
        name : name of the system
        config : dataset configuration dictionary

    Returns:
        Dataset for the system
    """
    get_dataset_fun = getattr(qk.datasets, f"get_{name}_dataset")
    return get_dataset_fun(**config)