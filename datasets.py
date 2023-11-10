import os
from torchvision import datasets, transforms
import torch
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform
def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'cifar10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    elif args.data_set == 'cifar100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
    elif args.data_set == "flowers":
        if is_train:
            train_dataset = datasets.Flowers102(root=args.data_path,
                                         split='train',
                                         download=True,
                                         transform=transform)
            val_dataset = datasets.Flowers102(root=args.data_path,
                                         split='val',
                                         download=True,
                                         transform=transform)
            dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
        else:
            dataset =  datasets.Flowers102(root=args.data_path,
                                     split='test',
                                     download=True,
                                     transform=transform)
        nb_classes = 102
    elif args.data_set == "pets":
        if is_train:
            dataset = datasets.OxfordIIITPet(root=args.data_path,
                                         split='trainval',
                                         download=True,
                                         transform=transform)
        else:
            dataset =  datasets.OxfordIIITPet(root=args.data_path,
                                     split='test',
                                     download=True,
                                     transform=transform)
        nb_classes = 37
    elif args.data_set == "stl10":
        if is_train:
            dataset = datasets.STL10(root=args.data_path,
                                         split='train',
                                         download=True,
                                         transform=transform)
        else:
            dataset = datasets.STL10(root=args.data_path,
                                     split='test',
                                     download=True,
                                     transform=transform)
        nb_classes = 10
    elif args.data_set == "food101":
        if is_train:
            dataset = datasets.Food101(root=args.data_path,
                                         split='train',
                                         download=True,
                                         transform=transform)
        else:
            dataset = datasets.Food101(root=args.data_path,
                                     split='test',
                                     download=True,
                                     transform=transform)
        nb_classes = 101
    elif args.data_set == "dtd":
        if is_train:
            train_dataset = datasets.DTD(root=args.data_path,
                                         split='train',
                                         download=True,
                                         transform=transform)
            val_dataset = datasets.DTD(root=args.data_path,
                                         split='val',
                                         download=True,
                                         transform=transform)
            dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
        else:
            dataset = datasets.DTD(root=args.data_path,
                                     split='test',
                                     download=True,
                                     transform=transform)
        nb_classes = 47
    elif args.data_set == "svhn":
        if is_train:
            dataset = datasets.SVHN(root=args.data_path,
                                         split='train',
                                         download=True,
                                         transform=transform)
        else:
            dataset = datasets.SVHN(root=args.data_path,
                                     split='test',
                                     download=True,
                                     transform=transform)
        nb_classes = 10
    elif args.data_set == "caltech101":
        dataset = datasets.Caltech101(root=args.data_path, download=True, transform=transform)
        trainset, testset = torch.utils.data.random_split(dataset, [0.7, 0.3],torch.Generator().manual_seed(0))
        dataset = trainset if is_train else testset
        nb_classes = 101
    elif args.data_set == "eurosat":
        dataset = datasets.ImageFolder(root=os.path.join(args.data_path, '2750'), transform=transform)
        trainset, testset = torch.utils.data.random_split(dataset, [0.7, 0.3],torch.Generator().manual_seed(0))
        dataset = trainset if is_train else testset
        nb_classes = 10
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)
    
    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if args.imagenet_default_mean_and_std:
        mean, std = (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    else:
        mean, std = (IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_MEAN)

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            no_aug = args.no_aug,
            hflip=args.hflip,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im and not args.no_aug:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:  
            t.append(
            transforms.Resize((args.input_size, args.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
