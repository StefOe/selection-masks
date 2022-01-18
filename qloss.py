import numpy as np
from torch import ones_like
from torch.nn.functional import interpolate


def qloss_f(hard_masks, weights):
    if len(hard_masks) == 0:
        return 1.0, []
    b_m = hard_masks[0].size(0)
    qloss = ones_like(hard_masks[0])
    qloss_parts = []
    prev_shape = (1, 1, 1)

    for hard_mask, weight in zip(hard_masks, weights):
        mask_shape = tuple(np.max([prev_shape, hard_mask.shape[2:]], 0))
        prev_shape = mask_shape
        # upscale mask and qloss if they don't match
        hard_mask = interpolate(hard_mask, size=mask_shape)
        qloss = interpolate(qloss, size=mask_shape)
        # calculate qloss
        hard_mask_loss = (hard_mask * weight).sum(1, keepdim=True)
        hard_mask_loss_ = hard_mask_loss.view(b_m, -1)
        qloss_parts.append((hard_mask_loss_.sum(1).mean()/hard_mask_loss_.size(1)).item())
        qloss *= hard_mask_loss

    # normalize to value from 0 to 1
    qloss = qloss.view(b_m, -1)
    qloss = qloss.sum(1).mean() / qloss.size(1)

    return qloss, qloss_parts
