import torch
from torch import nn as nn
from torch.nn.functional import interpolate

from utils import get_granularity_shape, DiscreteSigmoid, DiscreteSoftmax


class StaticAnyMask(nn.Module):
    def __init__(self, channel, height, width, granularity,
                 init_tau=1.0, input_drop_upscaling=True, init_mean=3, init_std=0.01):
        super(StaticAnyMask, self).__init__()
        self.tau = init_tau
        self.channel = channel
        self.width = width
        self.height = height
        self.granuality = granularity
        self.input_drop_upscaling = input_drop_upscaling
        self.init_mean = init_mean
        self.init_std = init_std
        self.discrete_sigmoid = DiscreteSigmoid()
        self.n_choices = 1

        # initialize any-selection
        c, w, h, upscale = get_granularity_shape(granularity, channel, height, width)
        self.upscale = upscale
        self.mask_shape = (c, self.height, self.width) if upscale else (c, h, w)
        self.n_mask_subpixel = c * w * h
        self.any_mask = nn.Parameter(torch.zeros(1, c, w, h, 1))
        self.reset_parameters()

    def reset_parameters(self):
        # initialize that the best resolutions takes full bits and the rest none
        torch.nn.init.normal_(self.any_mask[:, :, :, :, 0], self.init_mean, self.init_std)

    def forward(self, x):
        masked_images = x

        # apply any-selection
        hard_mask, soft_mask = self.discrete_sigmoid(self.any_mask, self.tau)
        hard_mask_ = hard_mask[:, 0]

        # upscale to apply to images (only for masks sizes bigger 1 and small original size):
        if self.upscale:
            hard_mask_ = interpolate(hard_mask_, size=(self.height, self.width))

        # mask images with given mask
        masked_images = masked_images * hard_mask_

        if self.input_drop_upscaling:
            upscale_ratio = ((hard_mask.detach().sum() + 1e-7) / self.n_mask_subpixel)
            masked_images = masked_images / upscale_ratio

        return masked_images, hard_mask, soft_mask


class StaticXorMask(nn.Module):
    def __init__(self, channel, height, width, n_choices, granularity, weights,
                 init_tau=1.0, init_std=0.01):
        super(StaticXorMask, self).__init__()
        self.weights = weights
        self.n_choices = n_choices
        self.granularity = granularity
        self.tau = init_tau
        self.channel = channel
        self.width = width
        self.height = height
        self.init_std = init_std
        self.discrete_softmax = DiscreteSoftmax()

        # initialize xor-selection
        c, w, h, upscale = get_granularity_shape(granularity, channel, height, width)
        self.upscale = upscale
        self.mask_shape = (c, self.height, self.width) if upscale else (c, h, w)
        mask = nn.Parameter(torch.zeros(1, c, h, w, n_choices))
        self.mask = mask
        self.reset_parameters()

    def reset_parameters(self):
        # initialize that the best featuremap takes full bits and the rest none
        for i in range(self.n_choices):
            init_m = self.weights[i] * self.n_choices
            torch.nn.init.normal_(self.mask[:, :, :, :, i], init_m, self.init_std)

    def forward(self, x):
        masked_images = x

        # ### apply xor-selection
        hard_mask, soft_mask = self.discrete_softmax(self.mask,  self.tau)

        # upscale to apply to images (only for masks sizes bigger 1 and small original size):
        hard_mask_ = interpolate(hard_mask, size=self.mask_shape) if self.upscale else hard_mask

        # detect if more xor operations are suggested by the data
        n_merges = len(x.shape) - 5
        for i in range(n_merges):
            hard_mask_ = hard_mask_.unsqueeze(2)

        # mask image
        masked_images = masked_images * hard_mask_
        # merge operation
        masked_images = masked_images.sum(1)

        return masked_images, hard_mask, soft_mask


class StaticMaskSequential(nn.Sequential):
    r"""A sequential container for selection masks.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    It will return the masked x (without altering it, just selecting), a list of the discrete masks ,
    and a list of the soft masks.


    To make it easier to understand, here is a small example::
        channel = 3
        width = 32
        height = 32
        n_choices = 2

        # Example of using Sequential
        model = StaticMaskSequential(
                  XorMask(channel, height, width, n_choices, "version", [1, 0.5]),
                  AnyMask(channel, height, width, "subpixel"),
                )

        # Example of using Sequential with OrderedDict
        model = StaticMaskSequential(OrderedDict([
                  ('Xor jpeg', XorMask(channel, height, width, n_choices, "version", [1, 0.5]),
                  ('Any subpixel', AnyMask(channel, height, width, "subpixel"),
                ]))

        x = torch.rand(1, n_choices, channel, width)
        masked_x, hard_masks, soft_masks = model(x)
    """

    def forward(self, x):
        hard_masks = []
        soft_masks = []
        for i, module in enumerate(self):
            x, hard_mask, soft_mask = module(x)
            hard_masks.append(hard_mask)
            soft_masks.append(soft_mask)
        return x, hard_masks, soft_masks