import torch.nn as nn
import os

from train.config.env_config import ACT_FUNC_NEG_SLOPE


class CustomLeakyReLU(nn.LeakyReLU):
    """
    Custom implementation of LeakyReLU
    """

    def __init__(self):
        super().__init__(negative_slope=ACT_FUNC_NEG_SLOPE)


def make_output_dirs(run_id: str, root_dir: str) -> tuple[str, str, str]:
    """
    Create output directories for a training run.
    """
    tensorboard_dir = f"{root_dir}/tensorboard/{run_id}"
    models_dir = f"{root_dir}/models/{run_id}"
    videos_dir = f"{root_dir}/videos/{run_id}"

    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)

    return tensorboard_dir, models_dir, videos_dir
