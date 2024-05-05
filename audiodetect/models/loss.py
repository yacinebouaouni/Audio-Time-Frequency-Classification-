from torch import nn


def get_loss(config):
    if config.LOSS == "cross_entropy":
        return nn.CrossEntropyLoss()
