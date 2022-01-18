#!/usr/bin/env python
# coding: utf-8
import argparse
import json
from os import makedirs

import numpy as np
import torch

from train import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='mask selection experiments')


    def parse_bool(arg):
        arg = arg.lower()
        if 'true'.startswith(arg):
            return True
        elif 'false'.startswith(arg):
            return False
        else:
            raise ValueError()


    defaults = {
        "lr": 0.001,
        "lr_net": 0.001,
        "batch_size": 128,
        "tau_init": 1.,
        "tau_min": 1.,
        "tau_factor": 1.,
        "log_iter": None,
        "n_epochs": None,
        "lambda_patience": None,
        "lambda_factor": 1.25,
        "lambda_init": 1.,
        "lambda_threshold": 0.0025,
        "quality_levels": None,
        "thumb_h": None,
        "thumb_w": None,
        "dynamic_h": -1,
        "dynamic_w": -1,
        "dynamic_mask": "linear",
        "dynamic_learn_static": False,
        "dynamic_learn_dynamic": True,
        "use_any_mask": True,
        "thumb_granularity": "version",
        "any_granularity": "subpixel",
        "fmap_granularity": "version",
        "any_init_mean": 3.0,
        "random_num_del": 1,
        "random_change_iter": 1,
        "input_drop_upscaling": True,
        "checkpoint_qloss_interval": 0.05,
        "use_warmup_net": False,
        "use_warmup_mask": False,
    }


    def str_to_bool(value):
        if isinstance(value, bool):
            return value
        if value.lower() in {'false', 'f', '0', 'no', 'n'}:
            return False
        elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
            return True
        raise ValueError(f'{value} is not a valid boolean value')


    parser.add_argument('--dataset', type=str, required=True,
                        help='which dataset (available: "mnist", "fashion_mnist", "cifar10", "sanity", "svhn", "supernova", "remotesensing", "imagenet", "ships")')
    parser.add_argument('--mask-type', type=str, required=True,
                        help='which mask type (available: "static", "dynamic", "random", "given")')
    parser.add_argument('--batch-size', type=int, default=defaults["batch_size"],
                        help='x batch size for training')
    parser.add_argument('--log-iter', type=int, default=defaults["log_iter"],
                        help='defines after how many iteration logging occures (defaults to experiment settings)')
    parser.add_argument('--lr', type=float, default=defaults["lr"],
                        help='learning rate')
    parser.add_argument('--lr-net', type=float, default=defaults["lr_net"],
                        help='learning rate of the prediction network')
    parser.add_argument('--tau-init', type=float, default=defaults["tau_init"],
                        help='initial temperature tau')
    parser.add_argument('--tau-min', type=float, default=defaults["tau_min"],
                        help='minimal temperature tau')
    parser.add_argument('--tau-factor', type=float, default=defaults["tau_factor"],
                        help='factor rate of temperature tau per epoch (<=1)')
    parser.add_argument('--checkpoint-qloss-interval', type=float, default=defaults["checkpoint_qloss_interval"],
                        help='save best accuracy in specific qloss intervals (0-1)')
    parser.add_argument('--n-repeats', type=int, default=30,
                        help='how often is the experiment repeated')
    parser.add_argument('--n-epochs', type=int, default=defaults["n_epochs"],
                        help='total number of epochs for training (defaults to experiment settings)')
    parser.add_argument('--lambda-patience', type=int, default=defaults["lambda_patience"],
                        help='number of patience epochs before lambda adaption')
    parser.add_argument('--lambda-factor', type=float, default=defaults["lambda_factor"],
                        help='factor to increase lambda after no change occured')
    parser.add_argument('--lambda-init', type=float, default=defaults["lambda_init"],
                        help='initial lambda value')
    parser.add_argument('--lambda-threshold', type=float, default=defaults["lambda_threshold"],
                        help='threshold value for lambda scheduler')
    parser.add_argument('--quality-levels', nargs='+', default=defaults["quality_levels"],
                        help='the jpeg quality levels (defaults to experiment settings)')
    parser.add_argument('--fmap-granularity', type=str, default=defaults["fmap_granularity"],
                        help='on which granularity feauture maps (jpeg) should be selected(channel, version)')
    parser.add_argument('--thumb-w', nargs='+', default=defaults["thumb_w"],
                        help='the list of widths to take for the thumbnails')
    parser.add_argument('--thumb-h', nargs='+', default=defaults["thumb_h"],
                        help='the list of heights to take for the thumbnails')
    parser.add_argument('--thumb-granularity', type=str, default=defaults["thumb_granularity"],
                        help='on which granularity thumbnails should be selected (subpixel, pixel, channel, version)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--noise-mean', type=float, default=None,
                        help='only for sanity dataset: mean value of noise')
    parser.add_argument('--noise-std', type=float, default=None,
                        help='only for sanity dataset: standard deviation of noise')
    parser.add_argument('--exp-dir', type=str, default="./runs",
                        help='where to save the experiments')
    parser.add_argument('--use-any-mask', type=str_to_bool, nargs='?',
                        const=True, default=defaults["use_any_mask"],
                        help="if the any mask should be used at all")
    parser.add_argument('--input-drop-upscaling', type=str_to_bool, nargs='?',
                        const=True, default=defaults["input_drop_upscaling"],
                        help="if the drop mask should be regularized like dropout")
    parser.add_argument('--any-init-mean', type=float, default=defaults["any_init_mean"],
                        help='What is the initial any mask mean?')
    parser.add_argument('--slurm-friendly', type=str_to_bool, nargs='?',
                        const=True, default=False,
                        help="boolean to be slurm friendly (only one run per process)")
    parser.add_argument('--any-granularity', type=str, default=defaults["any_granularity"],
                        help='on which granularity subpixel should be dropped(subpixel, pixel, channel, version, subnondrant, nondrant, 5drant, CXLIVdrant, ...)')
    parser.add_argument('--dynamic-w', type=int, default=defaults["dynamic_w"],
                        help='only for dynamic masking: set to width of thumbnail analyzed by the server (default is full size)')
    parser.add_argument('--dynamic-h', type=int, default=defaults["dynamic_h"],
                        help='only for dynamic masking: set to height of thumbnail analyzed by the server (default is full size)')
    parser.add_argument('--dynamic-mask', type=str, default=defaults["dynamic_mask"],
                        help='which mask model to use (linear, linconv, nonlinear, conv, att')
    parser.add_argument('--dynamic-learn-static', type=str_to_bool, nargs='?',
                        default=defaults["dynamic_learn_static"],
                        help="if the static mask should be adapted as well")
    parser.add_argument('--dynamic-learn-dynamic', type=str_to_bool, nargs='?',
                        default=defaults["dynamic_learn_dynamic"],
                        help="if the dynamic mask should be adapted")
    parser.add_argument('--use-warmup-net', type=str_to_bool, nargs='?',
                        default=defaults["use_warmup_net"],
                        help="if the pretrained net + optimizer should be used")
    parser.add_argument('--use-warmup-mask', type=str_to_bool, nargs='?',
                        default=defaults["use_warmup_mask"],
                        help="if the pretrained mask + optimizer should be used")
    parser.add_argument('--warmup-dir', type=str,
                        default="models",
                        help="directory of warmup file")
    parser.add_argument('--random-num-del', type=int, default=defaults["random_num_del"],
                        help='how many choices to make per iteration')
    parser.add_argument('--random-change-iter', type=int, default=defaults["random_change_iter"],
                        help='how long to wait until the next random change happens')
    parser.add_argument('--given-mask-path', type=str, default="mask.npy",
                        help='path for mask if "given" is used as mask')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.quality_levels is not None:
        args.quality_levels = [int(i) for i in args.quality_levels]
    if args.thumb_w is not None:
        args.thumb_w = [int(i) for i in args.thumb_w]
    if args.thumb_h is not None:
        args.thumb_h = [int(i) for i in args.thumb_h]

    exp_settings = {
        "batch_size": args.batch_size,
        "quality_levels": args.quality_levels,
        "thumb_w": args.thumb_w,
        "thumb_h": args.thumb_h,
    }

    if args.dataset.lower() == "mnist":
        from experiments import get_MNIST as get_data
    elif args.dataset.lower() == "fashion_mnist":
        from experiments import get_fashion_MNIST as get_data
    elif args.dataset.lower() in ["cifar10", "cifar"]:
        from experiments import get_cifar10 as get_data
    elif args.dataset.lower() == "sanity":
        defaults["noise_mean"] = 0.
        defaults["noise_std"] = 0.
        if args.noise_mean is None:
            noise_mean = defaults["noise_mean"]
        else:
            noise_mean = args.noise_mean
        if args.noise_std is None:
            noise_std = defaults["noise_std"]
        else:
            noise_std = args.noise_std
        exp_settings.update({
            "noise_std": noise_std,
            "noise_mean": noise_mean
        })
        from experiments import get_sanity as get_data
    elif args.dataset.lower() == "svhn":
        from experiments import get_svhn as get_data
    elif args.dataset.lower() == "ships":
        from experiments import get_ships as get_data
    elif args.dataset.lower() in ["galaxy", "galaxy10"]:
        from experiments import get_galaxy10 as get_data
    elif args.dataset.lower() == "remotesensing":
        from experiments import get_remotesensing as get_data
    else:
        raise Exception(
            f"Unknown dataset: {args.dataset} (available: 'mnist', 'fashion_mnist', 'cifar10', 'sanity', 'svhn', 'galaxy10', 'remotesensing').")
    experiment_params, dataset_defaults = get_data(**exp_settings)

    # create name depending on diverging defaults
    non_default_keys = []
    for key in defaults:
        default_val = dataset_defaults[key] if defaults[key] is None else defaults[key]
        if args.__dict__[key] is not None and default_val != args.__dict__[key]:
            non_default_keys.append(key)

    if len(non_default_keys) == 0:
        name = "default"
    else:
        name = ""
        for key in np.sort(non_default_keys):
            name += f" {key}-{str(args.__dict__[key]).replace(' ', '').replace('[','').replace(']','')}"

    # save experiment settings and args as json
    def to_json(dict, f_name):
        json_f = json.dumps(dict)
        f = open(f"{args.exp_dir}/{experiment_params['name'].lower()}/{args.mask_type}/{name}/{f_name}.json", "w")
        f.write(json_f)
        f.close()

    exp_params_save = experiment_params.copy()
    exp_params_save.pop("train_loader")
    exp_params_save.pop("val_loader")
    exp_params_save.pop("test_loader")
    exp_params_save["loss_f"] = exp_params_save["loss_f"].__class__.__name__
    exp_params_save.pop("network")

    try:
        makedirs(f"{args.exp_dir}/{experiment_params['name'].lower()}/{args.mask_type}/{name}")
    except FileExistsError:
        # directory already exists
        pass
    to_json(exp_params_save, "experiment")
    to_json(args.__dict__, "args")

    for run_i in range(args.n_repeats):
        run_name = f"{name}/run_{run_i}"
        # if run folder exists, we assume it is already done or running right now
        try:
            makedirs(rf"{args.exp_dir}/{experiment_params['name'].lower()}/{args.mask_type}/{run_name}")
        except FileExistsError:
            continue

        trainer = Trainer(run_name, args, experiment_params)
        net, mask, checkpoints= trainer.train()

        if args.slurm_friendly:
            break  # do only one run to allow easier parallelization (slurm friendly)
