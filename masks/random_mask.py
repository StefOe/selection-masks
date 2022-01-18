import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import choice
from numpy import arange
from masks.static_mask import get_granularity_shape


class RandomAnyMask(nn.Module):
    def __init__(self, channel, height, width, granularity,
                 init_tau=1.0, input_drop_upscaling=True, num_del=1, change_iter=1):
        super(RandomAnyMask, self).__init__()
        self.change_iter = change_iter
        self.tau = init_tau
        self.channel = channel
        self.width = width
        self.height = height
        self.granularity = granularity

        self.num_del = num_del
        self.input_drop_upscaling = input_drop_upscaling
        c, w, h, upscale = get_granularity_shape(granularity, channel, height, width)
        self.upscale = upscale

        self.mask_shape = (c, h, w)
        self.img_shape = (c, self.height, self.width) if upscale else (c, h, w)
        self.n_mask_subpixel = c * w * h
        any_mask = arange(c*w*h).tolist()
        self.any_mask = any_mask
        self.train_iter = 1

        self.param_placeholder = nn.Linear(1, 1)

    def forward(self, x):
        masked_images = x

        hard_mask, soft_mask = self._get_discrete_mask(x.device, *self.mask_shape)
        hard_mask_ = hard_mask[:, 0]

        # upscale to apply to images (only for masks sizes bigger 1 and small original size):
        if self.upscale:
            hard_mask_ = F.interpolate(hard_mask_, size=self.img_shape[1:])

        # mask images with given mask
        masked_images = masked_images * hard_mask_

        if self.input_drop_upscaling:
            upscale_ratio = ((hard_mask.detach().sum() + 1e-7) / self.n_mask_subpixel)
            masked_images = masked_images / upscale_ratio

        return masked_images, hard_mask, soft_mask

    def _get_discrete_mask(self, device, c, w, h):

        if self.training:
            # only flip num_del at a time
            if self.train_iter % self.change_iter == 0:
                for i in range(self.num_del):
                    len_retain = len(self.any_mask)
                    if len_retain == 0:
                        break
                    idx = choice(arange(len_retain))
                    del self.any_mask[idx]
            self.train_iter += 1

        mask = torch.zeros(c*w*h).to(device)
        mask[self.any_mask] = 1.0
        mask = mask.view(1, c, w, h, 1)
        hard_mask, soft_mask = mask, mask

        hard_mask = hard_mask.permute(0, 4, 1, 2, 3)
        return hard_mask, soft_mask