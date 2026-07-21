import numpy as np

from clintrials.core.registry import CORE_REGISTRY


def logit1(x, a0=3, beta=0):
    beta = np.clip(beta, CORE_REGISTRY["math_clip_beta_min"], CORE_REGISTRY["math_clip_beta_max"])
    return 1 / (1 + np.exp(-a0 - np.exp(beta) * x))

def inverse_logit1(x, a0=3, beta=0):
    beta = np.clip(beta, CORE_REGISTRY["math_clip_beta_min"], CORE_REGISTRY["math_clip_beta_max"])
    return (np.log(x / (1 - x)) - a0) / np.exp(beta)
