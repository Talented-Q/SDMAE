# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data == 'Tiny-ImageNet':
        root = os.path.join(args.data_path, 'train') if is_train else os.path.join(args.data_path, 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        args.nb_classes = 200
    elif args.data == 'CIFAR-100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        args.nb_classes = 100
    elif args.data == 'CIFAR-10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform)
        args.nb_classes = 10
    elif args.data == 'SVHN':
        option = 'train' if is_train else 'test'
        dataset = datasets.SVHN(args.data_path, split=option, transform=transform)
        args.nb_classes = 10
    elif args.data == 'APTOS':
        root = os.path.join(args.data_path, 'train') if is_train else os.path.join(args.data_path, 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        args.nb_classes = 5
    elif args.data == 'COVID':
        root = os.path.join(args.data_path, 'train') if is_train else os.path.join(args.data_path, 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        args.nb_classes = 2
    elif args.data == 'Image_folder':
        root = os.path.join(args.data_path, 'train') if is_train else os.path.join(args.data_path, 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
    else:
        raise Exception('wrong data mode')

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

class TransformTwice:
    def __init__(self, s_transform, w_transform):
        self.s_transform = s_transform
        self.w_transform = w_transform
    def __call__(self, inp):
        s_out = self.s_transform(inp)
        w_out = self.w_transform(inp)
        return [s_out, w_out]