from functools import partial

import torch
import torch.nn as nn
from torch._six import inf

EPS = 1e-8


class DiscreteSigmoid(nn.Module):
    def discrete(self, logits: torch.Tensor, tau: float, hard: bool):
        r"""
        Args:
          logits: `[batch_size, *shape]` weight output of mask
          tau: non-negative scalar temperature
          hard: if ``True``, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd

        Returns:
          If ``hard=True``, the returned samples will be one-hot, otherwise they will
          be probability distributions that sum to 1 across features

        """

        logits = logits / tau
        y_soft = torch.sigmoid(logits)
        if hard:
            y_hard = y_soft.round()
            # this cool bit of code achieves two things:
            # - makes the output value exactly one-hot (since we add then
            #   subtract y_soft value)
            # - makes the gradient equal to y_soft gradient (since we strip
            #   all other gradients)
            y = y_hard - y_soft.detach() + y_soft
        else:
            y = y_soft
        return y, y_soft

    def forward(self, mask: torch.Tensor, tau: float = 1., hard: bool = True):
        hard_mask, soft_mask = self.discrete(mask, tau=tau, hard=hard)
        hard_mask = hard_mask.permute(0, 4, 1, 2, 3)
        return hard_mask, soft_mask


class DiscreteSoftmax(nn.Module):
    def discrete(self, logits: torch.Tensor, tau: float = 1., hard: bool = True, n_k: int = 1):
        r"""
        Args:
          logits: `[batch_size, *shape]` weight output of mask
          tau: non-negative scalar temperature
          hard: if ``True``, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd

        Returns:
          If ``hard=True``, the returned samples will be one-hot, otherwise they will
          be probability distributions that sum to 1 across features

        """
        shape = logits.shape
        logits = logits / tau
        y_soft = logits.reshape(-1, shape[-1]).softmax(-1)
        if hard:
            k = y_soft.argsort(-1, descending=True)[:, :n_k]
            # this bit is based on
            # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
            y_hard = torch.zeros_like(y_soft).scatter_(-1, k, 1.0)
            # this cool bit of code achieves two things:
            # - makes the output value exactly one-hot (since we add then
            #   subtract y_soft value)
            # - makes the gradient equal to y_soft gradient (since we strip
            #   all other gradients)
            y = y_hard - y_soft.detach() + y_soft
        else:
            y = y_soft
        y = y.reshape(shape[0], shape[1], shape[2], shape[3], shape[4])
        return y, y_soft

    def forward(self, mask: torch.Tensor, tau: float = 1., hard: bool = True):
        hard_mask, soft_mask = self.discrete(mask, tau=tau, hard=hard)
        hard_mask = hard_mask.permute(0, 4, 1, 2, 3)
        return hard_mask, soft_mask


class LambdaSchedule:
    """
    adapted version of https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html
    """

    def __init__(self, initial_lambda, goal=0.0, mode='min', factor=1.25, patience=1,
                 verbose=False, threshold=0.0001, threshold_mode='abs',
                 cooldown=0, max_lambda=inf, eps=1e-8):
        self.initial_lambda = initial_lambda
        self.lambda_val = initial_lambda
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.goal = goal
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.eps = eps
        self.last_epoch = -1
        self.max_lambda = max_lambda

        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()
        self.cooldown_counter = cooldown

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._adapt_lambda(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            return True
        return False

    def _adapt_lambda(self, epoch):
        old_lambda = self.lambda_val
        new_lambda = min(old_lambda * self.factor, self.max_lambda)
        if abs(new_lambda - old_lambda) > self.eps:
            self.lambda_val = new_lambda
            if self.verbose:
                print('Epoch {:5d}: increasing lambda to {:.4e}.'.format(epoch, new_lambda))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _cmp(self, mode, threshold_mode, threshold, a, best):
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            return abs(self.goal - a) < abs(self.goal - best) * rel_epsilon

        elif mode == 'min' and threshold_mode == 'abs':
            return abs(self.goal - a) < abs(self.goal - best) - threshold

        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            return abs(self.goal - a) > abs(self.goal - best) * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return abs(self.goal - a) > abs(self.goal - best) + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)


def get_granularity_shape(granularity, c, w, h):
    upscale = False
    if granularity == "subpixel":
        c, h, w = c, h, w
    elif granularity == "pixel":
        c, h, w = 1, h, w
    elif granularity == "subquadrant":
        c, h, w = c, 2, 2
        upscale = True
    elif granularity == "quadrant":
        c, h, w = 1, 2, 2
        upscale = True
    elif granularity == "subnondrant":
        c, h, w = c, 3, 3
        upscale = True
    elif granularity == "nondrant":
        c, h, w = 1, 3, 3
        upscale = True
    elif granularity == "sub5drant":
        c, h, w = c, 5, 5
        upscale = True
    elif granularity == "5drant":
        c, h, w = 1, 5, 5
        upscale = True
    elif granularity == "subCXLIVdrant":
        c, h, w = c, 12, 12
        upscale = True
    elif granularity == "CXLIVdrant":
        c, h, w = 1, 12, 12
        upscale = True
    elif granularity == "sub48drant":
        c, h, w = c, 48, 48
        upscale = True
    elif granularity == "48drant":
        c, h, w = 1, 48, 48
        upscale = True
    elif granularity == "quadsubpixel":
        c, h, w = c, h // 4, w // 4
        upscale = True
    elif granularity == "quadpixel":
        c, h, w = 1, h // 4, w // 4
        upscale = True
    elif granularity == "channel":
        c, h, w = c, 1, 1
    elif granularity == "version":
        c, h, w = 1, 1, 1
    else:
        raise Exception(
            f"Chosen granularity '{granularity}' does not exist ('subpixel', 'pixel', 'channel', 'version')")

    return c, w, h, upscale


def icnr_init(weight, scale=2, init=torch.nn.init.kaiming_normal_):
    dim1, dim2, h, w = weight.shape
    ni2 = int(dim1 / (scale ** 2))
    k = init(torch.zeros([ni2, dim2, h, w])).transpose(0, 1)
    k = k.reshape(ni2, dim2, -1)
    k = k.repeat(1, 1, scale ** 2)
    k = k.reshape([dim2, dim1, h, w]).transpose(0, 1)
    weight.data.copy_(k)


class MiniUNet(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.in_size = in_size
        self.maxpool = nn.MaxPool2d(2)
        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)
        num_dims = [8, 16, 32, 64]
        self.down1 = self.get_u_block(in_size, num_dims[0])
        self.down2 = self.get_u_block(num_dims[0], num_dims[1])
        self.down3 = self.get_u_block(num_dims[1], num_dims[2])
        self.across = self.get_u_block(num_dims[2], num_dims[3])
        self.upscale1 = self.get_upscale(num_dims[3], num_dims[2], 2)
        self.up1 = self.get_u_block(num_dims[2] + num_dims[2], num_dims[2])
        self.upscale2 = self.get_upscale(num_dims[2], num_dims[1], 2)
        self.up2 = self.get_u_block(num_dims[1] + num_dims[1], num_dims[1])
        self.upscale3 = self.get_upscale(num_dims[1], num_dims[0], 2)
        self.up3 = self.get_u_block(num_dims[0] + num_dims[0], num_dims[0])

    def get_upscale(self, in_size, out_size, scale=2):
        upscale = nn.Sequential(
            nn.Conv2d(in_size, out_size * (scale ** 2), 1),
            self.relu,
            nn.PixelShuffle(2)
        )
        nn.utils.weight_norm(upscale[0])
        return upscale

    def get_u_block(self, in_size, hidden_size):
        block = nn.Sequential(
            self.pad,
            nn.Conv2d(in_size, hidden_size, 3),
            self.relu,
            self.pad,
            nn.Conv2d(hidden_size, hidden_size, 3),
            self.relu,
        )
        nn.utils.weight_norm(block[1])
        nn.utils.weight_norm(block[-2])
        return block

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(self.maxpool(down1))
        down3 = self.down3(self.maxpool(down2))

        across = self.across(self.maxpool(down3))

        up1_scale = self.upscale1(across)
        up1 = self.up1(torch.cat([up1_scale, down3], 1))

        up2_scale = self.upscale2(up1)
        up2 = self.up2(torch.cat([up2_scale, down2], 1))

        up3_scale = self.upscale3(up2)
        up3 = self.up3(torch.cat([up3_scale, down1], 1))

        return torch.cat([up3, x], 1)

    def reset_parameters(self):
        icnr_init(self.upscale1[0].weight)
        icnr_init(self.upscale2[0].weight)
        icnr_init(self.upscale3[0].weight)
