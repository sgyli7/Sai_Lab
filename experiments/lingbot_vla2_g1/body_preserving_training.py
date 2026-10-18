"""TRAIN-only loss weighting; native observations, queries and inference stay fixed."""
import torch

from .contract import ACTION_SLOTS, ACTION_MASK
from .runtime import Runtime

BODY_SLOTS = tuple(int(x) for x in ACTION_SLOTS[28:32])
LOSS_WEIGHTING = {'schema': 'lingbot_TRAIN_body4_weighted_native_mask_v1',
                  'effective32_body_indices': [28, 29, 30, 31],
                  'native55_body_slots': list(BODY_SLOTS), 'body4_weight': 4.0,
                  'other_valid_weight': 1.0, 'padded_or_unused_weight': 0.0,
                  'TRAIN_only': True, 'original_validation_unit_weights': True}
OPTIMIZATION_RESERVE_SECONDS = 180
VALIDATION_RESERVE_SECONDS = 90
DIAGNOSTIC_RESERVE_SECONDS = 45


def weighted_training_mask(mask):
    if mask.ndim != 3 or mask.shape[1:] != (50, 55):
        raise ValueError('Native batch/50/55 loss mask required')
    if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
        raise ValueError('Unweighted finite native mask required; no double weighting')
    valid = torch.as_tensor(ACTION_MASK, device=mask.device)
    if torch.any(mask[..., ~valid] != 0):
        raise ValueError('Unused native action slots must remain masked')
    weights = torch.ones(55, device=mask.device, dtype=torch.float32)
    weights[list(BODY_SLOTS)] = LOSS_WEIGHTING['body4_weight']
    return mask.to(torch.float32) * weights


class BodyPreservingRuntime(Runtime):
    def loss(self, rgb, state, normalized_actions, mask, *, train_weighted=False):
        if not train_weighted:
            return super().loss(rgb, state, normalized_actions, mask)
        inputs = self.inputs(rgb, state, normalized_actions, mask)
        inputs['joint_mask'] = weighted_training_mask(inputs['joint_mask'])
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == 'cuda'):
            result = self.model(**inputs)
        return result[0], result[1]
