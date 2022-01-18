#!/usr/bin/env python
# coding: utf-8
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import OrderedDict
from warnings import warn

from masks.given_mask import GivenAnyMask
from masks.static_mask import StaticXorMask, StaticAnyMask, StaticMaskSequential
from masks.random_mask import RandomAnyMask
from masks.dynamic_mask import DynamicAnyMask, DynamicXorMask, DynamicMaskSequential

from utils import LambdaSchedule
from qloss import qloss_f
from logger import Logger


class Trainer:
    def __init__(self, run_name, args, exp_params, test_loader="test", use_tensorboard=True):
        self.run_name = run_name
        self.args = args
        self.exp_params = exp_params
        self.use_tensorboard = use_tensorboard

        # use test or validation dataset (hyperparameters)
        if test_loader == "test":
            self.test_loader = self.exp_params["test_loader"]
        elif test_loader == "val":
            self.test_loader = self.exp_params["val_loader"]
        else:
            raise Exception("test_loader must either be 'test' or 'val'")

        self.n_iter = 1
        self.best_epoch = -1  # w.r.t. best qloss
        self.n_subpixel = exp_params["n_subpixel"]
        self.height, self.width = self.exp_params["width"], self.exp_params["height"]
        self.channel = self.exp_params["channel"]

        self.weights = []

        # ### init xor-selection variables (thumbnails) ###
        self.n_thumbs = self.exp_params["n_thumbs"]
        self.use_thumbs = self.n_thumbs > 0

        if self.use_thumbs:
            self.thumb_h, self.thumb_w = self.exp_params["thumb_h"], self.exp_params["thumb_w"]
            self.heights, self.widths = [self.height] + self.thumb_h, [self.width] + self.thumb_w
            self.size_reductions = [1.] + [(h + w) / (self.width + self.height) for h, w in
                                           zip(self.thumb_h, self.thumb_w)]
            self.thumb_weights = torch.tensor(self.size_reductions)
            self.weights.append(self.thumb_weights)

        # ### init xor-selection variables (thumbnails) ###
        self.n_featuremaps = self.exp_params["n_featuremaps"]
        self.use_fmaps = self.n_featuremaps > 1

        if self.use_fmaps:
            self.fmap_weights = []
            # calculate weight of each jpeg quality level for loss function
            # (e.g., 100 = 1, 50 = .5, 10 = .1)
            self.quality_weights = [1.]
            self.base_quality = 100
            for self.quality in self.exp_params["quality_levels"]:
                self.quality_weights.append(self.quality / self.base_quality)
            self.quality_weights = torch.tensor(self.quality_weights)
            if len(self.quality_weights) > 0:
                self.fmap_weights.append(self.quality_weights)

            self.fmap_weights = torch.cat(self.fmap_weights, 0)
            self.weights.append(self.fmap_weights)

        self.use_any_mask = args.use_any_mask
        if self.use_any_mask:
            self.weights.append(torch.tensor([1.]))

        # ### create mask + network (from experiment_params, which is a dict containing all objects needed) ###
        self._init_networks()

        # use standard devices (change with "export CUDA_VISIBLE_DEVICES=1")
        if self.args.cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.mask = self.mask.to(self.device)
        self.net = self.net.to(self.device)
        self.weights = [weight.view(1, -1, 1, 1, 1).to(self.device) for weight in self.weights]

        # loss function for learning task (e.g., cross entropy)
        self.criterion = self.exp_params["loss_f"]

        # optimizer (e.g., Adam)
        self._init_optimizer()


        # get cooldown timer (number of epochs from tau-init to tau-min)
        self.cooldown = 0
        tau = self.args.tau_init
        while tau > self.args.tau_min:
            tau *= self.args.tau_factor
            self.cooldown += 1
        self.tau = self.args.tau_init

        # ### lambda optimizer (based on learning rate scheduler) ###
        self.lambda_schedule = LambdaSchedule(
            initial_lambda=self.args.lambda_init, factor=self.args.lambda_factor,
            patience=self.args.lambda_patience if self.args.lambda_patience is not None else self.exp_params[
                "lambda_patience"],
            threshold=self.args.lambda_threshold,
            cooldown=self.cooldown,
            verbose=False
        )
        self.lamb = self.lambda_schedule.lambda_val

        if args.use_warmup_net or args.use_warmup_mask:
            try:
                save = torch.load(f"{args.warmup_dir}/{exp_params['name']}.pt", map_location=self.device)
                if args.use_warmup_net:
                    self.net.load_state_dict(save["net"])
                    self.net_optimizer.load_state_dict(save["net_optimizer"])
                if args.use_warmup_mask:
                    self.mask.load_state_dict(save["mask"])
                    self.mask_optimizer.load_state_dict(save["mask_optimizer"])
            except FileNotFoundError:
                warn(Warning(f"Warmup file '{exp_params['name']}.pt' cannot be found in '{args.warmup_dir}'. "
                              f"I will ignore it."))
        # ### logging ###
        self.log_iter = self.args.log_iter if self.args.log_iter is not None else self.exp_params["log_iter"]
        self.logdir = f"{self.args.exp_dir}/{self.exp_params['name']}/{self.args.mask_type}/{self.run_name}"
        if self.use_tensorboard:
            self.logger = Logger(
                log_iter=self.log_iter,
                use_any_mask=self.use_any_mask, use_thumbs=self.use_thumbs, use_fmap=self.use_fmaps,
                n_thumbs=self.n_thumbs, n_featuremaps=self.n_featuremaps,
                img_mean=self.exp_params["mean"], img_std=self.exp_params["std"], device=self.device,
                log_dir=self.logdir
            )
            self.logger.add_scalar("lambda", self.lamb, 0)
            self.logger.add_scalar("tau", self.tau, 0)

        # ### checkpointer
        self.checkpoint_qloss_interval = args.checkpoint_qloss_interval
        self.checkpoints = OrderedDict()

        # after how many iterations logs should written
        self.n_epochs = self.args.n_epochs if self.args.n_epochs is not None else self.exp_params["n_epochs"]


        # use jit to make it a tiny bit faster (only possible atm for models that operate on a single GPU)
        self.mask = torch.jit.script(self.mask) if self.args.mask_type.lower() in ["static", "dynamic"] else self.mask
        if not isinstance(self.net, torch.nn.DataParallel):
            self.net = torch.jit.script(self.net)


    def _init_networks(self):
        params = (
            self.channel, self.height, self.width,
        )

        masks = []
        if self.args.mask_type.lower() == "static":
            if self.use_thumbs:
                masks.append(StaticXorMask(
                    *params, self.n_thumbs + 1, self.args.thumb_granularity,
                    self.thumb_weights**2, self.args.tau_init
                ))
            if self.use_fmaps:
                masks.append(StaticXorMask(
                    *params, self.exp_params["n_featuremaps"], self.args.fmap_granularity,
                    self.fmap_weights**2, self.args.tau_init
                ))
            if self.use_any_mask:
                masks.append(StaticAnyMask(
                    *params, self.args.any_granularity, self.args.tau_init,
                    self.args.input_drop_upscaling, self.args.any_init_mean
                ))
            mask = StaticMaskSequential(*masks)
        elif self.args.mask_type.lower() == "random":
            if self.use_any_mask:
                masks.append(RandomAnyMask(
                    *params, self.args.any_granularity,
                    self.args.tau_init, self.args.input_drop_upscaling,
                    self.args.random_num_del, self.args.random_change_iter
                ))
            mask = StaticMaskSequential(*masks)
        elif self.args.mask_type.lower() == "given":
            if self.use_any_mask:
                masks.append(GivenAnyMask(
                    self.args.given_mask_path,
                    *params, self.args.any_granularity,
                    self.args.tau_init, self.args.input_drop_upscaling,
                    self.args.random_num_del, self.args.random_change_iter
                ))
            mask = StaticMaskSequential(*masks)
        elif self.args.mask_type.lower() == "dynamic":
            server_height, server_width = self.args.dynamic_h, self.args.dynamic_w
            if self.args.dynamic_h == -1:
                server_height = self.exp_params["height"]
            if self.args.dynamic_w == -1:
                server_width = self.exp_params["width"]

            params_suffix = (
                server_width, server_height, self.args.dynamic_mask,
                self.args.dynamic_learn_static, self.args.dynamic_learn_dynamic, self.args.tau_init
            )
            if self.use_thumbs:
                masks.append(DynamicXorMask(
                    *params, self.n_thumbs + 1, self.args.thumb_granularity,
                    self.thumb_weights**2, *params_suffix
                ))
            if self.use_fmaps:
                masks.append(DynamicXorMask(
                    *params, self.exp_params["n_featuremaps"], self.args.fmap_granularity,
                    self.fmap_weights**2, *params_suffix
                ))
            if self.use_any_mask:
                masks.append(DynamicAnyMask(
                    *params, self.args.any_granularity, *params_suffix,
                    self.args.input_drop_upscaling, self.args.any_init_mean
                ))
            mask = DynamicMaskSequential(*params, server_height, server_width, *masks)
        else:
            raise Exception(f"Mask type '{self.args.mask_type}' does not exist ('deconv', 'naive').")

        self.mask = mask
        self.net = self.exp_params["network"]()

    def _init_optimizer(self):
        self.net_optimizer = optim.AdamW(
            [{"params": self.net.parameters(), "weight_decay": 1e-4}], lr=self.args.lr_net,
            amsgrad=True, betas=(.5, .999))
        self.mask_optimizer = optim.AdamW(
            [{"params": self.mask.parameters(), "weight_decay": 1e-7}], lr=self.args.lr,
            amsgrad=True, betas=(.5, .999))

    def evaluate(self, test_loader=None):
        if test_loader is None:
            test_loader = self.test_loader

        correct = 0
        total = 0
        qloss_eval = 0
        qloss_parts = []
        self.mask.eval()
        self.net.eval()
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                masked_images, hard_masks, soft_masks = self.mask(images)

                outputs_mask = self.net(masked_images)
                _, predicted_mask = torch.max(outputs_mask.data, 1)
                total += labels.size(0)
                correct += (predicted_mask == labels).sum()

                qloss, qloss_parts = qloss_f(hard_masks, self.weights)
                qloss_eval += qloss

        correct = correct.item()
        qloss_eval /= len(test_loader)
        if len(hard_masks) != 0:
            qloss_eval = qloss_eval.item()
        accuracy = correct / total
        return (
            accuracy, qloss_eval, qloss_parts,
            hard_masks,
            masked_images
        )

    def train(self, train_loader=None, val_loader=None):
        if train_loader is None:
            train_loader = self.exp_params["train_loader"]

        hard_masks = None
        # initial performance test
        # (e.g., get initial test accuracy)
        accuracy, eval_qloss, eval_qloss_parts, eval_hard_masks, example_images = self.evaluate(val_loader)

        # initial logging
        if self.use_tensorboard:
            self.logger.add_scalar("eval_qloss", eval_qloss, self.n_iter)
            self.logger.add_scalar("eval_accuracy", accuracy, self.n_iter)
            self.logger.log_images(False, eval_hard_masks, self.n_iter)
            if self.use_any_mask and self.args.input_drop_upscaling:
                    any_mask = eval_hard_masks[-1]
                    any_mask = any_mask.view(any_mask.size(0), -1)
                    example_images *= (any_mask.sum(1) / any_mask.size(1)).view(-1, 1, 1, 1)
            self.logger.add_example_images("example_image", example_images, self.n_iter)

        # iterate over epochs (one epoch=one pass over the data)
        for epoch in range(self.n_epochs):

            self.mask.train()
            self.net.train()
            # iterate over batches
            for i, data in enumerate(train_loader, 0):

                # get the inputs
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                # zero the parameter gradients
                self.net.zero_grad()
                self.mask.zero_grad()

                # ### forward + backward + optimize ###
                # get the masked x and selection masks
                masked_images, hard_masks, soft_masks = self.mask(images)

                # use the masked x in the prediction network
                outputs_mask = self.net(masked_images)

                # compute the network loss
                loss = self.criterion(outputs_mask, labels)

                # compute qloss
                qloss, qloss_parts = qloss_f(hard_masks, self.weights)

                # join objectives
                comb_loss = loss + self.lamb * qloss
                comb_loss.backward()

                # optimize mask
                self.mask_optimizer.step()

                # optimize net
                self.net_optimizer.step()

                # log statistics per iteration
                if self.use_tensorboard:
                    self.logger.log_loss_tolist(loss.item(), comb_loss.item(), qloss.item(), qloss_parts, self.n_iter)

                self.n_iter += 1

            # decrease tau after each epoch
            if self.lambda_schedule.step(eval_qloss):
                # reset tau if lambda is increased
                for mask in self.mask:
                    mask.tau = self.args.tau_init
                self.lamb = self.lambda_schedule.lambda_val
            else:
                self.tau = np.max([self.tau * self.args.tau_factor, self.args.tau_min])
                for mask in self.mask:
                    mask.tau = self.tau

            # eval
            accuracy, eval_qloss, eval_qloss_parts, eval_hard_masks, example_images = self.evaluate(val_loader)

            # log epoch statistics
            if self.use_tensorboard:
                self.logger.add_scalar("eval_qloss", eval_qloss, self.n_iter)
                self.logger.add_scalar("eval_accuracy", accuracy, self.n_iter)
                self.logger.add_scalar("tau", self.tau, self.n_iter)
                self.logger.add_scalar("lambda", self.lamb, self.n_iter)
                self.logger.log_images(False, eval_hard_masks, self.n_iter)
                self.logger.log_images(True, hard_masks, self.n_iter)
                if self.use_any_mask and self.args.input_drop_upscaling:
                    any_mask = eval_hard_masks[-1]
                    any_mask = any_mask.view(any_mask.size(0), -1)
                    example_images *= (any_mask.sum(1) / any_mask.size(1)).view(-1, 1, 1, 1)
                self.logger.add_example_images("example_image", example_images, self.n_iter)

            # checkpoint
            interval = (eval_qloss // self.checkpoint_qloss_interval) * self.checkpoint_qloss_interval

            if (interval in self.checkpoints and self.checkpoints[interval]["accuracy"] < accuracy) or (
                    interval not in self.checkpoints):
                checkpoint = {
                    "accuracy": accuracy,
                    "qloss": eval_qloss,
                    "lambda": self.lamb,
                    "tau": self.tau,
                    "epoch": epoch,
                    "n_iter": self.n_iter - 1,
                }
                self.checkpoints[interval] = checkpoint
                checkpoint.update({
                    "net": self.net.state_dict(),
                    "net_optimizer": self.net_optimizer.state_dict(),
                    "mask": self.mask.state_dict(),
                    "mask_optimizer": self.mask_optimizer.state_dict(),
                })
                torch.save(checkpoint, f"{self.logdir}/checkpoint_{interval:.4f}.pt")
                if self.use_tensorboard:
                    qloss_checks = list(self.checkpoints.keys())
                    accuracy_checks = [self.checkpoints[key]["accuracy"] for key in qloss_checks]
                    f = plt.figure(1)
                    plt.plot(qloss_checks, accuracy_checks)
                    plt.xlim(0, 1); plt.ylim(0.8, 1)
                    plt.xlabel("Loss q"); plt.ylabel("Accuracy")
                    self.logger.add_figure("qloss vs accuracy", f, self.n_iter)

        if self.use_tensorboard:
            self.logger.close()

        return self.net, self.mask, self.checkpoints
