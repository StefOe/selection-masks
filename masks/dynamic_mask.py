from abc import ABC

import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn as nn

from utils import get_granularity_shape, MiniUNet, DiscreteSigmoid, DiscreteSoftmax


class DynamicAnyMask(nn.Module):
    def __init__(self, channel, height, width, granularity, server_height, server_width, mask_model,
                 learn_static=False, learn_dynamic=True, init_tau=1.0, input_drop_upscaling=True,
                 init_mean=3, init_std=0.01):
        super(DynamicAnyMask, self).__init__()
        self.server_height = server_height
        self.server_width = server_width
        self.init_mean = init_mean
        self.init_std = init_std
        self.granularity = granularity
        self.tau = init_tau
        self.channel = channel
        self.width = width
        self.height = height
        self.input_drop_upscaling = input_drop_upscaling
        self.discrete_sigmoid = DiscreteSigmoid()

        mask_ = get_mask(mask_model)

        # initialize any-selection
        c, w, h, upscale = get_granularity_shape(granularity, channel, width, height)
        self.upscale = upscale
        self.mask = mask_(
            channel, self.server_width, self.server_height, c, h, w, 1, [self.init_mean],
            learn_static, learn_dynamic, init_std
        )

        self.n_mask_subpixel = c * w * h

    def reset_parameters(self):
        self.mask.reset_paramters()

    def forward(self, x, server_x):
        masked_images = x

        # apply any-selection
        hard_mask, soft_mask = self.discrete_sigmoid(self.mask(server_x), self.tau)
        hard_mask_ = hard_mask[:, 0]

        # upscale to apply to images (only for masks sizes bigger 1 and small original size):
        if self.upscale:
            hard_mask_ = F.interpolate(hard_mask_, size=(self.height, self.width))

        # mask images with given mask
        masked_images = masked_images * hard_mask_

        if self.input_drop_upscaling:
            b = x.size(0)
            masked_images = masked_images / ((hard_mask.detach().reshape(b, -1).sum(1).view(
                b, 1, 1, 1) + 1e-7) / self.n_mask_subpixel)

        return masked_images, hard_mask, soft_mask


class DynamicXorMask(nn.Module):
    def __init__(self, channel, height, width,
                 n_choices, granularity, weights,
                 server_height, server_width, mask_model,
                 learn_static=False, learn_dynamic=True, init_tau=1.0, init_std=0.01):
        super(DynamicXorMask, self).__init__()
        self.server_height = server_height
        self.server_width = server_width
        self.weights = weights
        self.n_choices = n_choices
        self.granularity = granularity
        self.tau = init_tau
        self.channel = channel
        self.width = width
        self.height = height
        self.discrete_softmax = DiscreteSoftmax()

        mask_ = get_mask(mask_model)

        # initialize xor-selection
        c, h, w, upscale = get_granularity_shape(granularity, channel, height, width)
        self.upscale = upscale
        self.mask_shape = (c, self.height, self.width) if upscale else (c, h, w)
        mask = mask_(
            channel, self.server_height, self.server_width,
            c, h, w, n_choices, weights, learn_static, learn_dynamic, init_std
        )
        self.mask = mask

    def reset_parameters(self):
        self.mask.reset_parameters()

    def forward(self, x, server_x):
        masked_images = x

        # ### apply xor-selection
        hard_mask, soft_mask = self.discrete_softmax(self.mask(server_x), self.tau)

        # upscale to apply to images (only for masks sizes bigger 1 and small original size):
        hard_mask_ = F.interpolate(hard_mask, size=self.mask_shape) if self.upscale else hard_mask

        # detect if more xor operations are suggested by the data
        n_merges = len(x.shape) - 5
        for i in range(n_merges):
            hard_mask_ = hard_mask_.unsqueeze(2)

        # mask image
        masked_images = masked_images * hard_mask_
        # merge operation
        masked_images = masked_images.sum(1)

        return masked_images, hard_mask, soft_mask


class DynamicMaskSequential(nn.Sequential):
    r"""A sequential container for selection masks.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    It will return the masked x (without altering it, just selecting), a list of the discrete masks ,
    and a list of the soft masks.


    To make it easier to understand, here is a small example::
        channel = 3
        width = 32
        height = 32
        server_width = 32
        server_height = 32
        n_choices = 2

        # Example of using Sequential
        model = StaticMaskSequential(
                  DynamicXorMask(channel, height, width, n_choices, "version", [1, 0.5], server_height, server_width),
                  Dynamic AnyMask(channel, height, width, "subpixel", server_height, server_width),
                )

        # Example of using Sequential with OrderedDict
        model = StaticMaskSequential(OrderedDict([
                  ('Xor jpeg', DynamicXorMask(
                    channel, height, width, n_choices, "version", [1, 0.5], server_height, server_width
                  ),
                  ('Any subpixel', DynamicAnyMask(channel, height, width, "subpixel", server_height, server_width),
                ]))

        x = torch.rand(1, n_choices, channel, height, width)
        masked_x, hard_masks, soft_masks = model(x)
    """

    def __init__(self, channel, height, width, server_height, server_width, *args):
        super(DynamicMaskSequential, self).__init__(*args)
        self.server_thumb = server_height != height and server_width != width
        self.server_height = server_height
        self.server_width = server_width
        self.mask_shape = (channel, width, height)

    def forward(self, x):
        server_x = x.view(x.size(0), -1, *self.mask_shape)[:, 0]
        if self.server_thumb:
            server_x = F.adaptive_max_pool2d(server_x, (self.server_height, self.server_width))

        hard_masks = []
        soft_masks = []
        for module in self:
            x, hard_mask, soft_mask = module(x, server_x)
            hard_masks.append(hard_mask)
            soft_masks.append(soft_mask)
        return x, hard_masks, soft_masks


def get_mask(mask_model):
    if mask_model == "linear":
        mask_ = _LinearMask
    elif mask_model == "mlp":
        mask_ = _MLPMask
    elif mask_model == "conv":
        mask_ = _ConvMask
    elif mask_model == "convatt":
        mask_ = _ConvAttMask
    elif mask_model == "smallconvatt":
        mask_ = _SmallConvAttMask
    elif mask_model == "convnet":
        mask_ = _ConvNetMask
    elif mask_model == "unet":
        mask_ = _UNetMask
    elif mask_model == "att":
        mask_ = _SelfAttentionMask
    else:
        raise Exception(f"mask model for dynamic mask unavailable: {mask_model}")
    return mask_


def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class _ConvModelMask(nn.Module, ABC):
    def __init__(self,
                 in_channel, in_height, in_width,
                 out_channel, out_height, out_width, out_choices,
                 init_weights, learn_static, learn_dynamic, init_std, bias=False):
        super(_ConvModelMask, self).__init__()
        self.bias = bias
        self.in_channel = in_channel
        self.out_channel, self.out_height, self.out_width = out_channel, out_height, out_width
        self.out_choices = out_choices
        self.init_weights = init_weights
        self.learn_dynamic = learn_dynamic
        self.init_std = init_std

        if learn_static:
            self.mask = nn.Parameter(torch.zeros(1, out_channel, out_height, out_width, out_choices),
                                     requires_grad=True)
        else:
            self.mask = nn.Parameter(torch.zeros(1, 1, 1, 1, out_choices), requires_grad=False)

    def set_learn_dynamic(self, choice):
        if not choice:
            for p in self.conv.parameters():
                p.requires_grad = False

    def reset_parameters(self):
        # initialize that the best featuremap takes full bits and the rest none
        for fmap_out in range(self.out_choices):
            init_m = self.init_weights[fmap_out] * self.out_choices
            init.normal_(self.mask[:, :, :, :, fmap_out], init_m, self.init_std)

        self.conv.apply(init_weights)
        if not self.learn_dynamic:
            init.constant_(self.conv[-1].weight, 0)
        else:
            init.xavier_normal_(self.conv[-1].weight, gain=0.001)

    def forward(self, input):
        b = input.size(0)
        out = self.conv(input)
        out_w, out_h = out.shape[2], out.shape[3]
        if out_w != self.out_width or out_h != self.out_height:
            if out_w > self.out_width and out_h > self.out_height:
                out = F.adaptive_avg_pool2d(out, output_size=(self.out_height, self.out_width))
            else:
                out = F.interpolate(out, size=(self.out_height, self.out_width))
        out = out.view(b, self.out_choices, self.out_channel, self.out_height, self.out_width).permute(0, 2, 3, 4, 1)
        out += self.mask
        return out


class _UNetMask(_ConvModelMask):

    def __init__(self,
                 in_channel, in_height, in_width,
                 out_channel, out_height, out_width, out_choices,
                 init_weights, learn_static, learn_dynamic, init_std, bias=False):
        super(_UNetMask, self).__init__(
            in_channel, in_height, in_width, out_channel, out_height, out_width, out_choices,
            init_weights, learn_static, learn_dynamic, init_std, bias
        )
        self.conv = nn.Sequential(
            MiniUNet(in_channel),
            nn.Conv2d(8 + in_channel, self.out_choices * self.out_channel, 1, bias=self.bias)
        )
        self.set_learn_dynamic(learn_dynamic)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.conv[0].reset_parameters()


class _ConvNetMask(_ConvModelMask):

    def __init__(self,
                 in_channel, in_height, in_width,
                 out_channel, out_height, out_width, out_choices,
                 init_weights, learn_static, learn_dynamic, init_std, bias=False):
        super(_ConvNetMask, self).__init__(
            in_channel, in_height, in_width, out_channel, out_height, out_width, out_choices,
            init_weights, learn_static, learn_dynamic, init_std, bias
        )

        layers = []
        in_dims = in_channel
        dims = 8
        h, w = in_height, in_width
        while h > out_height or w > out_width:
            layers.append(Block(in_dims, dims))
            h //= 2
            w //= 2
            in_dims = dims
            dims *= 2
        layers.append(Block(in_dims, dims, stride=1))

        layers.append(nn.Conv2d(dims, self.out_choices * self.out_channel, 1, bias=bias))
        self.conv = nn.Sequential(*layers)
        self.set_learn_dynamic(learn_dynamic)
        self.reset_parameters()


class GlobalAttention(nn.Module):
    def __init__(self, dims):
        super(GlobalAttention, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(dims, dims, 1)

    def forward(self, x, y):
        att = torch.sigmoid(self.conv(self.pool(x + y)))
        return x * att + y * (1 - att)


class ConvAttBlock(nn.Module):
    def __init__(self, in_dims, hidden_dims):
        super(ConvAttBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.pad2 = nn.ReflectionPad2d(2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dims, hidden_dims, 3),
            nn.BatchNorm2d(hidden_dims),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dims, hidden_dims, 3, dilation=2),
            nn.BatchNorm2d(hidden_dims),
            nn.ReLU(inplace=True),
        )
        self.att = GlobalAttention(hidden_dims)

    def forward(self, x):
        out_1 = self.conv1(self.pad1(x))
        out_2 = self.conv3(self.pad2(x))
        out = self.att(out_1, out_2)
        return out


class _ConvAttMask(_ConvModelMask):

    def __init__(self,
                 in_channel, in_height, in_width,
                 out_channel, out_height, out_width, out_choices,
                 init_weights, learn_static, learn_dynamic, init_std, bias=False):
        super(_ConvAttMask, self).__init__(
            in_channel, in_height, in_width, out_channel, out_height, out_width, out_choices,
            init_weights, learn_static, learn_dynamic, init_std, bias
        )

        dims = 8

        self.conv = nn.Sequential(
            ConvAttBlock(in_channel, dims),
            nn.Conv2d(dims, self.out_choices * self.out_channel, 1, bias=False)
        )
        self.set_learn_dynamic(learn_dynamic)

        self.reset_parameters()


class SmallConvAttBlock(nn.Module):
    def __init__(self, in_dims, hidden_dims):
        super(SmallConvAttBlock, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dims, hidden_dims, 1),
            nn.BatchNorm2d(hidden_dims),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dims, hidden_dims, 3),
            nn.BatchNorm2d(hidden_dims),
            nn.ReLU(inplace=True),
        )
        self.att = GlobalAttention(hidden_dims)

    def forward(self, x):
        out_1 = self.conv1(x)
        out_2 = self.conv3(self.pad(x))
        out = self.att(out_1, out_2)
        return out


class _SmallConvAttMask(_ConvModelMask):

    def __init__(self,
                 in_channel, in_height, in_width,
                 out_channel, out_height, out_width, out_choices,
                 init_weights, learn_static, learn_dynamic, init_std, bias=False):
        super(_SmallConvAttMask, self).__init__(
            in_channel, in_height, in_width, out_channel, out_height, out_width, out_choices,
            init_weights, learn_static, learn_dynamic, init_std, bias
        )

        dims = 8

        self.conv = nn.Sequential(
            SmallConvAttBlock(in_channel, dims),
            nn.Conv2d(dims, self.out_choices * self.out_channel, 1, bias=False)
        )
        self.set_learn_dynamic(learn_dynamic)

        self.reset_parameters()


class Block(nn.Module):
    def __init__(self, in_size, hidden_size, stride=2):
        super(Block, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pad = nn.ReflectionPad2d(1)
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(in_size, hidden_size, 3, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_size)

        if in_size != hidden_size:
            self.downsample = [
                nn.Conv2d(in_size, hidden_size, 1, stride=stride, bias=False),
                nn.BatchNorm2d(hidden_size)
            ]
            self.downsample = nn.Sequential(*self.downsample)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(self.pad(x))
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(self.pad(out))
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class _ConvMask(_ConvModelMask):

    def __init__(self,
                 in_channel, in_height, in_width,
                 out_channel, out_height, out_width, out_choices,
                 init_weights, learn_static, learn_dynamic, init_std, bias=False):
        super(_ConvMask, self).__init__(
            in_channel, in_height, in_width, out_channel, out_height, out_width, out_choices,
            init_weights, learn_static, learn_dynamic, init_std, bias
        )
        self.hidden_dim = 64
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.in_channel, self.out_choices * self.out_channel, 3, bias=self.bias),
        )
        self.set_learn_dynamic(learn_dynamic)
        self.reset_parameters()


class _MLPMask(nn.Module):
    def __init__(self,
                 in_channel, in_height, in_width,
                 out_channel, out_height, out_width, out_choices,
                 init_weights, learn_static, learn_dynamic, init_std, bias=False):
        super(_MLPMask, self).__init__()
        self.out_channel, self.out_height, self.out_width = out_channel, out_height, out_width
        self.out_choices = out_choices
        self.init_weights = init_weights
        self.learn_dynamic = learn_dynamic
        self.init_std = init_std
        in_n_feat = in_channel * in_width * in_height
        self.in_n_feat = in_n_feat
        self.out_n_feat = out_channel * out_width * out_height
        hidden_dim = 2 ** (3 * out_choices)
        self.weight1 = nn.Parameter(torch.zeros(hidden_dim, self.in_n_feat))
        self.weight2 = nn.Parameter(torch.zeros(self.out_n_feat * out_choices, hidden_dim))
        self.bias = bias
        self.bias1 = nn.Parameter(torch.zeros(hidden_dim))
        if bias:
            self.bias2 = nn.Parameter(torch.zeros(self.out_n_feat * out_choices))
        else:
            self.register_parameter('bias2', None)
        if not learn_dynamic:
            self.weight1.requires_grad = False
            self.weight2.requires_grad = False
            self.bias1.requires_grad = False
            if bias:
                self.bias2.requires_grad = False
        if learn_static:
            self.mask = nn.Parameter(torch.zeros(1, out_channel, out_height, out_width, out_choices),
                                     requires_grad=True)
        else:
            self.mask = nn.Parameter(torch.zeros(1, 1, 1, 1, out_choices), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        # initialize that the best featuremap takes full bits and the rest none
        for fmap_out in range(self.out_choices):
            init_m = self.init_weights[fmap_out] * self.out_choices
            init.normal_(self.mask[:, :, :, :, fmap_out], init_m, self.init_std)

        init.xavier_normal_(self.weight1)
        if self.learn_dynamic:
            init.xavier_normal_(self.weight2, 0.0001)
        else:
            init.zeros_(self.weight2)

        init.zeros_(self.bias1)
        if self.bias:
            init.zeros_(self.bias2)

    def forward(self, input):
        b = input.size(0)
        out = F.relu_(F.linear(input.view(b, -1), self.weight1, self.bias1))
        out = F.linear(out, self.weight2, self.bias2)
        out = out.view(b, self.out_channel, self.out_height, self.out_width, self.out_choices)
        out += self.mask
        return out


class _LinearMask(nn.Module):
    def __init__(self,
                 in_channel, in_height, in_width,
                 out_channel, out_height, out_width, out_choices,
                 init_weights, learn_static, learn_dynamic, init_std, bias=False):
        super(_LinearMask, self).__init__()
        in_n_feat = in_channel * in_width * in_height
        self.in_n_feat = in_n_feat
        self.out_channel, self.out_height, self.out_width = out_channel, out_height, out_width
        self.out_choices = out_choices
        self.init_weights = init_weights
        self.init_std = init_std
        self.learn_dynamic = learn_dynamic
        self.out_n_feat = out_channel * out_width * out_height
        self.weight = nn.Parameter(torch.zeros(self.out_n_feat * out_choices, self.in_n_feat))
        self.gamma = nn.Parameter(torch.tensor([0.]), requires_grad=learn_dynamic)

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_n_feat * out_choices))
        else:
            self.register_parameter('bias', None)

        if not learn_dynamic:
            self.weight.requires_grad = False
            if bias:
                self.bias.requires_grad = False

        if learn_static:
            self.mask = nn.Parameter(torch.zeros(1, out_channel, out_height, out_width, out_choices),
                                     requires_grad=True)
        else:
            self.mask = nn.Parameter(torch.zeros(1, 1, 1, 1, out_choices), requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self):
        # initialize that the best featuremap takes full bits and the rest none
        for fmap_out in range(self.out_choices):
            init_m = self.init_weights[fmap_out] * self.out_choices
            init.normal_(self.mask[:, :, :, :, fmap_out], init_m, self.init_std)

        if self.learn_dynamic:
            init.xavier_normal_(self.weight)
        else:
            init.zeros_(self.weight)
        init.zeros_(self.gamma)

        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input):
        b = input.size(0)
        out = self.gamma * F.linear(input.view(b, -1), self.weight, self.bias)
        out = out.view(b, self.out_channel, self.out_height, self.out_width, self.out_choices)

        out += self.mask
        return out


def conv1d(ni: int, no: int, bias: bool):
    "Create and initialize a `nn.Conv1d` layer with spectral normalization."
    conv = nn.Conv1d(ni, no, 1, bias=False)
    return conv  # spectral_norm(conv)


class _SelfAttentionMask(nn.Module):
    "Self attention layer for nd."

    def __init__(self, in_channel, in_height, in_width,
                 out_channel, out_height, out_width, out_choices,
                 init_weights, learn_static, learn_dynamic, init_std, bias=False):
        super().__init__()
        in_n_feat = in_channel * in_width * in_height
        self.in_n_feat = in_n_feat
        self.out_channel, self.out_height, self.out_width = out_channel, out_height, out_width
        self.out_choices = out_choices
        self.init_weights = init_weights
        self.init_std = init_std
        self.learn_dynamic = learn_dynamic
        self.bias = bias
        self.query = conv1d(in_channel, in_channel, bias)
        self.key = conv1d(in_channel, in_channel, bias)
        self.value = conv1d(in_channel, self.out_choices * out_channel, bias)
        self.gamma = nn.Parameter(torch.tensor([0.]), requires_grad=learn_dynamic)

        if not learn_dynamic:
            self.query.weight.requires_grad = False
            self.key.weight.requires_grad = False
            self.value.weight.requires_grad = False
            if bias:
                self.query.bias.requires_grad = False
                self.key.bias.requires_grad = False
                self.value.bias.requires_grad = False

        if learn_static:
            self.mask = nn.Parameter(torch.zeros(1, out_channel, out_height, out_width, out_choices),
                                     requires_grad=True)
        else:
            self.mask = nn.Parameter(torch.zeros(1, 1, 1, 1, out_choices), requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self):
        # initialize that the best featuremap takes full bits and the rest none
        for fmap_out in range(self.out_choices):
            init_m = self.init_weights[fmap_out] * self.out_choices
            init.normal_(self.mask[:, :, :, :, fmap_out], init_m, self.init_std)

        if self.learn_dynamic:
            [init.xavier_normal_(w) for w in [self.query.weight, self.key.weight, self.value.weight]]
        else:
            [init.zeros_(w) for w in [self.query.weight, self.key.weight, self.value.weight]]
        init.zeros_(self.gamma)

        if self.bias:
            [init.zeros_(b) for b in [self.query.bias, self.key.bias, self.value.bias]]

    def forward(self, x):
        # Notation from https://arxiv.org/pdf/1805.08318.pdf
        b = x.size(0)
        size = x.size()
        x = x.view(size[0], size[1], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        out = self.gamma * torch.bmm(h, beta)
        out = out.reshape(size[0], -1, size[2], size[3])
        if size[2] != self.out_width or size[3] != self.out_height:
            if size[2] > self.out_width and size[3] > self.out_height:
                out = F.adaptive_avg_pool2d(out, output_size=(self.out_height, self.out_width))
            else:
                out = F.interpolate(out, size=(self.out_height, self.out_width))
        out = out.view(b, self.out_choices, self.out_channel, self.out_height, self.out_width).permute(0, 2, 3, 4, 1)
        out += self.mask
        return out
