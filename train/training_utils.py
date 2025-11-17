import torch.nn as nn

from train.config.env_config import ACT_FUNC_NEG_SLOPE


class CustomLeakyReLU(nn.LeakyReLU):
    """
    Custom implementation of LeakyReLU
    """

    def __init__(self):
        super().__init__(negative_slope=ACT_FUNC_NEG_SLOPE)
