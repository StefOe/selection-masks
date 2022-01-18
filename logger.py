from torch.utils.tensorboard import SummaryWriter
import torch


class Logger(SummaryWriter):
    def __init__(self, log_iter, use_any_mask, use_thumbs, use_fmap, n_featuremaps, n_thumbs, img_mean, img_std, device,
                 **kwargs):
        self.log_iter = log_iter
        self.n_featuremaps = n_featuremaps
        self.n_thumbs = n_thumbs
        self.use_fmap = use_fmap
        self.use_thumbs = use_thumbs
        self.use_any_mask = use_any_mask
        self.device = device
        self.img_mean = torch.tensor(img_mean).view(1, -1, 1, 1).to(device)
        self.img_std = torch.tensor(img_std).view(1, -1, 1, 1).to(device)
        super(Logger, self).__init__(**kwargs)
        self.lists = {}
        self.counts = {}

    def add_to_list(self, name: str, value):
        if name not in self.lists:
            self.lists[name] = torch.tensor(0.)
            self.counts[name] = 0.
        self.lists[name] += value
        self.counts[name] += 1

    def send_lists(self, n_iter):
        for key in self.lists:
            self.add_scalar(key, self.lists[key].item() / self.counts[key], n_iter)
            self.lists[key] = torch.tensor(0.)
            self.counts[key] = 0.

    def log_images(self, train: bool, masks: list, n_iter: int):
        prefix = "train" if train else "eval"
        part_i = 0
        if self.use_thumbs:
            size_mask = masks[part_i]
            for i in range(self.n_thumbs + 1):
                s_mask = size_mask[0, i]
                self.add_image(f"{prefix}_mask_size:{i}", s_mask, n_iter)
            part_i += 1
        if self.use_fmap:
            for fmap_i in range(self.n_featuremaps):
                fm_mask = masks[part_i][0, fmap_i]
                self.add_image(f"{prefix}_fmap:{fmap_i}", fm_mask, n_iter)
            part_i += 1
        if self.use_any_mask:
            use_mask = 1 - masks[part_i][0, 0]
            self.add_image(f"{prefix}_mask_any", use_mask, n_iter)

    def log_loss_tolist(self, loss, comb_loss, qloss, qloss_parts, n_iter):
        self.add_to_list("train loss", loss)
        self.add_to_list("train combined loss", comb_loss)
        qloss_part_i = 0
        if self.use_thumbs:
            self.add_to_list("train size_loss", qloss_parts[qloss_part_i])
            qloss_part_i += 1
        if self.use_fmap:
            self.add_to_list("train fmap_loss", qloss_parts[qloss_part_i])
            qloss_part_i += 1
        if self.use_any_mask:
            self.add_to_list("train any_loss", qloss_parts[qloss_part_i])
        if len(qloss_parts) > 0:
            self.add_to_list("train qloss", qloss)

        if n_iter % self.log_iter == 0:
            self.send_lists(n_iter)

    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        if img_tensor.shape[0] > 3:
            img_tensor = img_tensor.permute(1, 0, 2).contiguous().view(1, img_tensor.size(1),
                                                                       img_tensor.size(2) * img_tensor.size(0))
        super().add_image(tag, img_tensor, global_step, walltime, dataformats)

    def add_example_images(self, tag, img_tensor, global_step=None, walltime=None, num=3, normalize=True):
        if normalize:
            img_tensor *= self.img_std
            img_tensor += self.img_mean
        for i in range(num):
            self.add_image(f"{tag}_{i}", img_tensor[i], global_step, walltime)
