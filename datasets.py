import random

import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from torchvision.transforms.functional import crop, InterpolationMode

from timm.data import create_transform

from continual_datasets.continual_datasets import *

import utils
import math
from functools import partial

__all__ = ['build_continual_dataloader', 'get_dataset', 'build_upstream_continual_dataloader', 'build_transform',
           'build_cifar_transform', 'build_cgqa_transform']

class Lambda(transforms.Lambda):
    def __init__(self, lambd, nb_classes):
        super().__init__(lambd)
        self.nb_classes = nb_classes

    def __call__(self, img):
        return self.lambd(img, self.nb_classes)


def target_transform(x, nb_classes):
    return x + nb_classes


def build_continual_dataloader(args):
    dataloader = list()
    dataloader_per_cls = dict()
    class_mask = list() if args.task_inc or args.train_mask else None
    target_task_map = dict()

    if 'cifar' in args.dataset.lower():
        transform_train = build_cifar_transform(True, args)
        transform_val = build_cifar_transform(False, args)
    elif 'cgqa' in args.dataset.lower() or 'cobj' in args.dataset.lower():
        transform_train = build_cgqa_transform(True)
        transform_val = build_cgqa_transform(False)
    else:
        transform_train = build_transform(True, args)
        transform_val = build_transform(False, args)

    if args.dataset.startswith('Split-'):
        dataset_train, dataset_val = get_dataset(args.dataset.replace('Split-', ''), transform_train, transform_val,
                                                 args)
        dataset_train_mean, dataset_val_mean = get_dataset(args.dataset.replace('Split-', ''), transform_val,
                                                           transform_val, args)

        args.nb_classes = len(dataset_val.classes)

        splited_dataset, class_mask, target_task_map = split_single_dataset(dataset_train, dataset_val, args)
        splited_dataset_per_cls = split_single_class_dataset(dataset_train_mean, dataset_val_mean, class_mask, args)

    else:
        if args.dataset == '5-datasets':
            dataset_list = ['SVHN', 'MNIST', 'CIFAR10', 'NotMNIST', 'FashionMNIST']
        else:
            dataset_list = args.dataset.split(',')

        if args.shuffle:
            random.shuffle(dataset_list)
        print(dataset_list)

        args.nb_classes = 0
        splited_dataset_per_cls = {}

    for i in range(args.num_tasks):
        if args.dataset.startswith('Split-'):
            dataset_train, dataset_val = splited_dataset[i]

        else:
            if 'cifar' in dataset_list[i].lower():
                transform_train = build_cifar_transform(True, args)
                transform_val = build_cifar_transform(False, args)
            dataset_train, dataset_val = get_dataset(dataset_list[i], transform_train, transform_val, args)
            dataset_train_mean, dataset_val_mean = get_dataset(dataset_list[i], transform_val, transform_val, args)

            transform_target = Lambda(target_transform, args.nb_classes)

            if class_mask is not None and target_task_map is not None:
                class_mask.append([i + args.nb_classes for i in range(len(dataset_val.classes))])
                for j in range(len(dataset_val.classes)):
                    target_task_map[j + args.nb_classes] = i
                args.nb_classes += len(dataset_val.classes)

            if not args.task_inc:
                dataset_train.target_transform = transform_target
                dataset_val.target_transform = transform_target
                dataset_train_mean.target_transform = transform_target
                dataset_val_mean.target_transform = transform_target

            # print(class_mask[i])

            splited_dataset_per_cls.update(split_single_class_dataset(dataset_train_mean, dataset_val_mean, [class_mask[i]], args))

        if args.distributed and utils.get_world_size() > 1:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()

            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)

            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        dataloader.append({'train': data_loader_train, 'val': data_loader_val})

    for i in range(len(class_mask)):
        for cls_id in class_mask[i]:
            dataset_train_cls, dataset_val_cls = splited_dataset_per_cls[cls_id]

            if args.distributed and utils.get_world_size() > 1:
                num_tasks = utils.get_world_size()
                global_rank = utils.get_rank()

                sampler_train = torch.utils.data.DistributedSampler(
                    dataset_train_cls, num_replicas=num_tasks, rank=global_rank, shuffle=True)

                sampler_val = torch.utils.data.SequentialSampler(dataset_val_cls)
            else:
                sampler_train = torch.utils.data.RandomSampler(dataset_train_cls)
                sampler_val = torch.utils.data.SequentialSampler(dataset_val_cls)

            data_loader_train_cls = torch.utils.data.DataLoader(
                dataset_train_cls, sampler=sampler_train,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
            )

            data_loader_val_cls = torch.utils.data.DataLoader(
                dataset_val_cls, sampler=sampler_val,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
            )

            dataloader_per_cls[cls_id] = {'train': data_loader_train_cls, 'val': data_loader_val_cls}

    return dataloader, dataloader_per_cls, class_mask, target_task_map


def get_dataset(dataset, transform_train, transform_val, args, target_transform=None):
    if dataset == 'CIFAR100':
        dataset_train = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR100(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'CGQA':
        dataset_train = CGQA(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = CGQA(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'COBJ':
        dataset_train = COBJ(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = COBJ(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'CIFAR10':
        dataset_train = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'MNIST':
        dataset_train = MNIST_RGB(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = MNIST_RGB(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'FashionMNIST':
        dataset_train = FashionMNIST(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = FashionMNIST(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'SVHN':
        dataset_train = SVHN(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = SVHN(args.data_path, split='test', download=True, transform=transform_val)

    elif dataset == 'NotMNIST':
        dataset_train = NotMNIST(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = NotMNIST(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'Flower102':
        dataset_train = Flowers102(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = Flowers102(args.data_path, split='test', download=True, transform=transform_val)

    elif dataset == 'Cars196':
        dataset_train = StanfordCars(args.data_path, split='train', download=True, transform=transform_train, target_transform=target_transform).data
        dataset_val = StanfordCars(args.data_path, split='test', download=True, transform=transform_val, target_transform=target_transform).data

    elif dataset == 'CUB200':
        dataset_train = CUB200(args.data_path, train=True, download=True, transform=transform_train, target_transform=target_transform).data
        dataset_val = CUB200(args.data_path, train=False, download=True, transform=transform_val, target_transform=target_transform).data

    elif dataset == 'Scene67':
        dataset_train = Scene67(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = Scene67(args.data_path, train=False, download=True, transform=transform_val).data

    elif dataset == 'TinyImagenet':
        dataset_train = TinyImagenet(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = TinyImagenet(args.data_path, train=False, download=True, transform=transform_val).data

    elif dataset == 'Imagenet-R':
        dataset_train = Imagenet_R(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = Imagenet_R(args.data_path, train=False, download=True, transform=transform_val).data

    else:
        raise ValueError('Dataset {} not found.'.format(dataset))

    return dataset_train, dataset_val


def split_single_dataset(dataset_train, dataset_val, args):
    if 'cgqa' in args.dataset.lower():
        # big first task
        labels = [26, 86, 2, 55, 75, 93, 16, 73, 54, 95,
                  53, 92, 78, 13, 7, 30, 22, 24, 33, 8,
                  43, 62, 3, 71, 45, 48, 6, 99, 82, 76,
                  60, 80, 90, 68, 51, 27, 18, 56, 63, 74,
                  1, 61, 42, 41, 4, 15, 17, 40, 38, 5,
                  91, 59, 0, 34, 28, 50, 11, 35, 23, 52,
                  10, 31, 66, 57, 79, 85, 32, 84, 14, 89,
                  19, 29, 49, 97, 98, 69, 20, 94, 72, 77,
                  25, 37, 81, 46, 39, 65, 58, 12, 88, 70,
                  87,36, 21, 83, 9, 96, 67, 64, 47, 44]
        classes_per_task = [10 * (10 - args.num_tasks + 1), 10]
    # elif 'cobj' in args.dataset.lower():
    #     pass
    else:
        nb_classes = len(dataset_val.classes)
        # assert nb_classes % args.num_tasks == 0
        classes_per_task = math.ceil(nb_classes / args.num_tasks)

        labels = [i for i in range(nb_classes)]

        if args.shuffle:
            random.shuffle(labels)

    split_datasets = list()
    mask = list()

    target_task_map = {}

    for i in range(args.num_tasks):
        train_split_indices = []
        test_split_indices = []

        if type(classes_per_task) is list:
            cpt = classes_per_task[0 if i == 0 else 1]
        else:
            cpt = classes_per_task
        scope = labels[:cpt]
        labels = labels[cpt:]

        mask.append(scope)
        for k in scope:
            target_task_map[k] = i

        for k in range(len(dataset_train.targets)):
            if int(dataset_train.targets[k]) in scope:
                train_split_indices.append(k)

        for h in range(len(dataset_val.targets)):
            if int(dataset_val.targets[h]) in scope:
                test_split_indices.append(h)

        subset_train, subset_val = Subset(dataset_train, train_split_indices), Subset(dataset_val, test_split_indices)

        split_datasets.append([subset_train, subset_val])

    return split_datasets, mask, target_task_map


def split_single_class_dataset(dataset_train, dataset_val, mask, args):
    nb_classes = len(dataset_val.classes)
    print(nb_classes)
    split_datasets = dict()
    print(mask)
    for i in range(len(mask)):
        single_task_labels = mask[i]
        # print(single_task_labels)

        # if args.dataset.startswith('Split-'):
        #     cls_ids = single_task_labels
        # else:
        #     cls_ids = list(range(len(single_task_labels)))
        for cls_id in single_task_labels:
            train_split_indices = []
            test_split_indices = []

            for k in range(len(dataset_train.targets)):
                if int(dataset_train.targets[k]) == cls_id:
                    train_split_indices.append(k)
            # print(len(train_split_indices))

            for h in range(len(dataset_val.targets)):
                if int(dataset_val.targets[h]) == cls_id:
                    test_split_indices.append(h)

            subset_train, subset_val = Subset(dataset_train, train_split_indices), Subset(dataset_val,
                                                                                          test_split_indices)

            split_datasets[cls_id] = [subset_train, subset_val]

    return split_datasets


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())

    return transforms.Compose(t)


#def build_transform(is_train, args):
#    resize_im = args.input_size > 32
#    dset_mean = (0.0, 0.0, 0.0)
#    dset_std = (1.0, 1.0, 1.0)
#
#    if is_train:
#        transform = transforms.Compose([
#            transforms.RandomResizedCrop((args.input_size, args.input_size)),
#            transforms.RandomHorizontalFlip(),
#            transforms.ToTensor(),
#            transforms.Normalize(dset_mean, dset_std),
#        ])
#        return transform
#
#    t = []
#    if resize_im:
#        size = int((256 / 224) * args.input_size)
#        t.append(
#            transforms.Resize(size),  # to maintain same ratio w.r.t. 224 images
#        )
#    t.append(transforms.ToTensor())
#    t.append(transforms.Normalize(dset_mean, dset_std))

#    return transforms.Compose(t)


def build_cifar_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
        ])
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)))

    return transforms.Compose(t)


def build_cgqa_transform(is_train, img_size=(224, 224)):
    if is_train:
        _train_transform = create_transform(
            input_size=img_size,
            is_training=is_train,
            color_jitter=0.3,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
        )
        # replace RandomResizedCropAndInterpolation with Resize, for not cropping img and missing concepts
        _train_transform.transforms[0] = transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC)

        return _train_transform
    else:
        return _build_default_transform(img_size, False)


def _build_default_transform(image_size=(128, 228), is_train=True, normalize=True):
    """
    Default transforms borrowed from MetaShift.
    Imagenet normalization.
    """
    _train_transform = [
            transforms.Resize(image_size),  # allow reshape but not equal scaling
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
    ]
    _eval_transform = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
    ]
    if normalize:
        _train_transform.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ))
        _eval_transform.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ))

    _default_train_transform = transforms.Compose(_train_transform)
    _default_eval_transform = transforms.Compose(_eval_transform)

    if is_train:
        return _default_train_transform
    else:
        return _default_eval_transform


# This is used for few shot learning
def split_multiple_dataset(datasets_info, args):
    split_datasets = list()
    target_dataset_map = dict()
    target_task_map = dict()
    task_dataset_map = dict()
    mask = list()
    last_index = 0 
    num_tasks = 0
    last_task = 0
    for name, dataset in datasets_info.items():
        args.nb_classes += dataset['num_classes']
        num_tasks += dataset['num_tasks']
        max_classes_per_task = math.ceil(dataset['num_classes'] / dataset['num_tasks'])
        class_per_task = [max_classes_per_task for i in range(dataset['num_tasks'])]
        class_per_task[-1] = dataset['num_classes'] % max_classes_per_task if dataset['num_classes'] % max_classes_per_task != 0 else class_per_task[-1]
        labels = [i + last_index for i in range(dataset['num_classes'])]

        if args.shuffle:
            random.shuffle(labels)
        
        for i in range(dataset['num_tasks']):
            train_split_indices = []
            test_split_indices = []

            scope = labels[:class_per_task[i]]
            labels = labels[class_per_task[i]:]

            mask.append(scope)

            for k in range(len(dataset['train'].targets)):
                if int(dataset['train'].targets[k]) + last_index in scope:
                    train_split_indices.append(k)

            for h in range(len(dataset['val'].targets)):
                if int(dataset['val'].targets[h]) + last_index in scope:
                    test_split_indices.append(h)

            subset_train, subset_val = Subset(dataset['train'], train_split_indices), Subset(dataset['val'], test_split_indices)

            split_datasets.append([subset_train, subset_val])
            task_dataset_map[i + last_task] = name

        
        last_index += dataset['num_classes']
        last_task += dataset['num_tasks']

    print(mask)
    tasks = [i for i in range(num_tasks)]
    if args.shuffle:
        random.shuffle(tasks)

    shuffle_split_datasets = []
    shuffle_mask = []
    shuffle_task_dataset_map = dict()

    for i, task_id in enumerate(tasks):
        shuffle_split_datasets.append(split_datasets[task_id])
        shuffle_mask.append(mask[task_id])
        shuffle_task_dataset_map[i] = task_dataset_map[task_id]
        for k in mask[task_id]:
            target_task_map[k] = i
            target_dataset_map[k] = task_dataset_map[task_id]

    return shuffle_split_datasets, shuffle_mask, target_dataset_map, target_task_map, shuffle_task_dataset_map


def build_upstream_continual_dataloader(args):
    dataloader = list()
    dataloader_per_cls = dict()
    class_mask = list() if args.task_inc or args.train_mask else None
    args.nb_classes = 0
    args.num_datasets = len(args.datasets)
    args.num_tasks = sum(args.tasks_per_dataset)
    datasets_info = dict(dict())
    last_classes_index = 0

    for i, dataset in enumerate(args.datasets):
        if 'cifar' in dataset.lower():
            transform_train = build_cifar_transform(True, args)
            transform_val = build_cifar_transform(False, args)
        elif 'cgqa' in args.dataset.lower() or 'cobj' in args.dataset.lower():
            transform_train = build_cgqa_transform(True)
            transform_val = build_cgqa_transform(False)
        else:
            transform_train = build_transform(True, args)
            transform_val = build_transform(False, args)
        dataset_train, dataset_val = get_dataset(dataset.replace('Split-', ''), transform_train, transform_val,
                                                 args, target_transform=partial(target_transform, nb_classes=last_classes_index))
        # dataset_train_mean, dataset_val_mean = get_dataset(dataset.replace('Split-', ''), transform_val,
        #                                                    transform_val, args)

        datasets_info[i] = dict()
        datasets_info[i]['train'] = dataset_train
        datasets_info[i]['val'] = dataset_val
        datasets_info[i]['num_classes'] = len(args.continual_datasets_targets[i])
        datasets_info[i]['num_tasks'] = args.tasks_per_dataset[i]
        last_classes_index += datasets_info[i]['num_classes']

    splited_dataset, class_mask, target_dataset_map, target_task_map, task_dataset_map = split_multiple_dataset(datasets_info, args)


    for i in range(args.num_tasks):
        
        dataset_train, dataset_val = splited_dataset[i]

        if args.distributed and utils.get_world_size() > 1:
            num_replicas = utils.get_world_size()
            global_rank = utils.get_rank()
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_replicas, rank=global_rank, shuffle=True)

            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        dataloader.append({'train': data_loader_train, 'val': data_loader_val})
    
    return dataloader, class_mask, target_dataset_map, target_task_map, task_dataset_map
