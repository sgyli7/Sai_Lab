"""Prospective data-derived point-mass inverse for exactly constant channels.

The original fitted statistics, model weights, input normalization, mapping and
all physical criteria remain independently frozen. Only action dimensions with
strictly zero original TRAIN variance AND min=max=mean use their empirical mean.
This candidate must pass a matched development ablation before any task claim.
"""
import numpy as np

from .data import MeanStd
from .runtime import Runtime


class PointMassMeanStd(MeanStd):
    def __init__(self, path):
        super().__init__(path)
        stat = self.doc['actions']
        std = np.asarray(stat['raw_std'])
        low, high, mean = (np.asarray(stat[k]) for k in ('min', 'max', 'mean'))
        self.constant = (std == 0) & (low == high) & (low == mean)
        if self.constant.shape != (32,):
            raise ValueError('Original32-channel empirical TRAIN statistics required')

    def denormalize(self, values, kind):
        result = super().denormalize(values, kind)
        if kind == 'actions':
            result[..., self.constant] = np.asarray(self.doc[kind]['mean'], np.float32)[self.constant]
        return result


class PointMassRuntime(Runtime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalizer = PointMassMeanStd(self.norm_path)
