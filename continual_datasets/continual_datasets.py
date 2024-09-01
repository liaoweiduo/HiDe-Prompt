import os

import os.path
import pathlib
from pathlib import Path

from typing import Any, Tuple

import glob
from shutil import move, rmtree
import json

import numpy as np

import torch
from torchvision import datasets
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg, download_and_extract_archive
from torchvision import transforms
from torchvision.transforms.functional import crop, InterpolationMode
from tqdm import tqdm
from datetime import datetime

import PIL
from PIL import Image

from .dataset_utils import read_image_file, read_label_file


class CGQA(torch.utils.data.Dataset):
    """
    This class extends the basic Pytorch Dataset class to handle list of paths
    as the main data source.
    """
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.download_dataset()

        datasets, label_info = self._get_datasets(
            self.root, mode='continual', image_size=(224, 224),
            load_set='train' if self.train else 'test')

        train_set, val_set, test_set = datasets['train'], datasets['val'], datasets['test']
        (label_set, map_tuple_label_to_int, map_int_label_to_tuple, meta_info
         ) = label_info
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.label_set = label_info

        # get targets
        if train:
            self.targets = train_set.targets
        else:
            self.targets = test_set.targets

        self.classes = np.unique(self.targets)

    def __getitem__(self, index):
        if self.train:
            data = self.train_set[index]
        else:
            data = self.test_set[index]

        img, target, ori_idx = data

        return img, target

    def download_dataset(self):
        pass

    def _get_datasets(self, dataset_root, image_size=(128, 128), shuffle=False, seed=None,
                      mode='continual', num_samples_each_label=None, label_offset=0, load_set=None):
        """
        Create GQA dataset, with given json files,
        containing instance tuples with shape (img_name, label).

        You may need to specify label_offset if relative label do not start from 0.

        :param dataset_root: Path to the dataset root folder.
        :param image_size: size of image.
        :param shuffle: If true, the train sample order (in json)
            in the incremental experiences is
            randomly shuffled. Default to False.
        :param seed: A valid int used to initialize the random number generator.
            Can be None.
        :param mode: Option [continual, sys, pro, sub, non, noc, nons, syss].
        :param num_samples_each_label: If specify a certain number of samples for each label,
            random sampling (build-in seed:1234,
            and replace=True if num_samples_each_label > num_samples, else False)
            is used to sample.
            Only for continual mode, only apply to train dataset.
        :param label_offset: specified if relative label not start from 0.
        :param load_set: train -> only pre-load train set;
            val -> only pre-load val set;
            test -> only pre-load test set.
            default None.

        :return data_sets defined by json file and label information.
        """
        img_folder_path = os.path.join(dataset_root, "CFST", "CGQA", "GQA_100")

        def preprocess_concept_to_integer(img_info, mapping_tuple_label_to_int_concepts):
            for item in img_info:
                item['concepts'] = [mapping_tuple_label_to_int_concepts[concept] for concept in item['comb']]

        def preprocess_label_to_integer(img_info, mapping_tuple_label_to_int):
            for item in img_info:
                item['image'] = f"{item['newImageName']}.jpg"
                item['label'] = mapping_tuple_label_to_int[tuple(sorted(item['comb']))]
                for obj in item['objects']:
                    obj['image'] = f"{obj['imageName']}.jpg"

        def formulate_img_tuples(images):
            """generate train_list and test_list: list with img tuple (path, label)"""
            img_tuples = []
            for item in images:
                instance_tuple = (
                item['image'], item['label'], item['concepts'], item['position'])  # , item['boundingBox']
                img_tuples.append(instance_tuple)
            return img_tuples

        if mode == 'continual':
            train_json_path = os.path.join(img_folder_path, "continual", "train", "train.json")
            val_json_path = os.path.join(img_folder_path, "continual", "val", "val.json")
            test_json_path = os.path.join(img_folder_path, "continual", "test", "test.json")

            with open(train_json_path, 'r') as f:
                train_img_info = json.load(f)
            with open(val_json_path, 'r') as f:
                val_img_info = json.load(f)
            with open(test_json_path, 'r') as f:
                test_img_info = json.load(f)
            # img_info:
            # [{'newImageName': 'continual/val/59767',
            #   'comb': ['hat', 'leaves'],
            #   'objects': [{'imageName': '2416370', 'objName': 'hat',
            #                'attributes': ['red'], 'boundingBox': [52, 289, 34, 45]},...]
            #   'position': [4, 1]},...]

            '''preprocess labels to integers'''
            label_set = sorted(list(set([tuple(sorted(item['comb'])) for item in val_img_info])))
            # [('building', 'sign'), ...]
            map_tuple_label_to_int = dict((item, idx + label_offset) for idx, item in enumerate(label_set))
            # {('building', 'sign'): 0, ('building', 'sky'): 1, ...}
            map_int_label_to_tuple = dict((idx + label_offset, item) for idx, item in enumerate(label_set))
            # {0: ('building', 'sign'), 1: ('building', 'sky'),...}
            '''preprocess concepts to integers'''
            concept_set = sorted(list(set([concept for item in val_img_info for concept in item['comb']])))
            mapping_tuple_label_to_int_concepts = dict((item, idx) for idx, item in enumerate(concept_set))
            # 21 concepts {'bench': 0, 'building': 1, 'car': 2, ...}
            map_int_concepts_label_to_str = dict((idx, item) for idx, item in enumerate(concept_set))
            # 21 concepts {0: 'bench', 1: 'building', 2: 'car', ...}

            preprocess_label_to_integer(train_img_info, map_tuple_label_to_int)
            preprocess_label_to_integer(val_img_info, map_tuple_label_to_int)
            preprocess_label_to_integer(test_img_info, map_tuple_label_to_int)

            preprocess_concept_to_integer(train_img_info, mapping_tuple_label_to_int_concepts)
            preprocess_concept_to_integer(val_img_info, mapping_tuple_label_to_int_concepts)
            preprocess_concept_to_integer(test_img_info, mapping_tuple_label_to_int_concepts)

            '''if num_samples_each_label provided, sample images to balance each class for train set'''
            selected_train_images = []
            if num_samples_each_label is not None and num_samples_each_label > 0:
                imgs_each_label = dict()
                for item in train_img_info:
                    label = item['label']
                    if label in imgs_each_label:
                        imgs_each_label[label].append(item)
                    else:
                        imgs_each_label[label] = [item]
                build_in_seed = 1234
                build_in_rng = np.random.RandomState(seed=build_in_seed)
                for label, imgs in imgs_each_label.items():
                    selected_idxs = build_in_rng.choice(
                        np.arange(len(imgs)), num_samples_each_label,
                        replace=True if num_samples_each_label > len(imgs) else False)
                    for idx in selected_idxs:
                        selected_train_images.append(imgs[idx])
            else:
                selected_train_images = train_img_info

            '''generate train_list and test_list: list with img tuple (path, label)'''
            train_list = formulate_img_tuples(selected_train_images)
            val_list = formulate_img_tuples(val_img_info)
            test_list = formulate_img_tuples(test_img_info)
            # [('continual/val/59767.jpg', 0),...

            '''shuffle the train set'''
            if shuffle:
                rng = np.random.RandomState(seed=seed)
                order = np.arange(len(train_list))
                rng.shuffle(order)
                train_list = [train_list[idx] for idx in order]

            '''generate train_set and test_set using PathsDataset'''
            train_set = self.PathsDataset(
                root=img_folder_path,
                files=train_list,  # train_list,      val_list for debug
                transform=transforms.Compose([transforms.Resize(image_size)]),
                loaded=load_set == 'train',
                name='con_train',
            )
            val_set = self.PathsDataset(
                root=img_folder_path,
                files=val_list,
                transform=transforms.Compose([transforms.Resize(image_size)]),
                loaded=load_set == 'val',
                name='con_val',
            )
            test_set = self.PathsDataset(
                root=img_folder_path,
                files=test_list,
                transform=transforms.Compose([transforms.Resize(image_size)]),
                loaded=load_set == 'test',
                name='con_test',
            )

            datasets = {'train': train_set, 'val': val_set, 'test': test_set}
            meta_info = {
                "concept_set": concept_set,
                "mapping_tuple_label_to_int_concepts": mapping_tuple_label_to_int_concepts,
                "map_int_concepts_label_to_str": map_int_concepts_label_to_str,
                "train_list": train_list, "val_list": val_list, "test_list": test_list}
            label_info = (label_set, map_tuple_label_to_int, map_int_label_to_tuple, meta_info)

        elif mode in ['sys', 'pro', 'sub', 'non', 'noc', 'nons', 'syss']:
            json_name = \
            {'sys': 'sys/sys_fewshot.json', 'pro': 'pro/pro_fewshot.json', 'sub': 'sub/sub_fewshot.json',
             'non': 'non_novel/non_novel_fewshot.json', 'noc': 'non_comp/non_comp_fewshot.json'}[mode]
            json_path = os.path.join(img_folder_path, "fewshot", json_name)
            with open(json_path, 'r') as f:
                img_info = json.load(f)
            label_set = sorted(list(set([tuple(sorted(item['comb'])) for item in img_info])))
            map_tuple_label_to_int = dict((item, idx + label_offset) for idx, item in enumerate(label_set))
            map_int_label_to_tuple = dict((idx + label_offset, item) for idx, item in enumerate(label_set))
            preprocess_label_to_integer(img_info, map_tuple_label_to_int)
            img_list = formulate_img_tuples(img_info)
            dataset = self.PathsDataset(
                root=img_folder_path,
                files=img_list,
                transform=transforms.Compose([transforms.Resize(image_size)]),
                loaded=True,
                name=f'few_{mode}',
            )

            datasets = {'dataset': dataset}
            meta_info = {"img_list": img_list}
            label_info = (label_set, map_tuple_label_to_int, map_int_label_to_tuple, meta_info)

        else:
            raise Exception(f'Un-implemented mode "{mode}".')

        return datasets, label_info

    class PathsDataset(torch.utils.data.Dataset):
        """
        This class extends the basic Pytorch Dataset class to handle list of paths
        as the main data source.
        """

        @staticmethod
        def default_image_loader(path):
            return Image.open(path).convert("RGB")

        def __init__(
                self,
                root,
                files,
                transform=None,
                target_transform=None,
                loader=default_image_loader,
                loaded=True,
                name='data',
        ):
            """
            Creates a File Dataset from a list of files and labels.

            :param root: root path where the data to load are stored. May be None.
            :param files: list of tuples. Each tuple must contain two elements: the
                full path to the pattern and its class label. Optionally, the tuple
                may contain a third element describing the bounding box to use for
                cropping (top, left, height, width).
            :param transform: eventual transformation to add to the input data (x)
            :param target_transform: eventual transformation to add to the targets
                (y)
            :param loader: loader function to use (for the real data) given path.
            :param loaded: True, load images into memory.
            If False, load when call getitem.
            Default True.
            :param name: Name if save to folder
            """

            if root is not None:
                root = Path(root)

            self.root = root
            self.imgs = files
            self.targets = [img_data[1] for img_data in self.imgs]
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader
            self.loaded = loaded
            self.name = name

            if self.loaded:
                self.load_data()

        def load_data(self):
            """
            load all data and replace imgs.
            """
            print(f'[{datetime.now().strftime("%Y/%m/%d %H:%M:%S")}] Load data in PathsDataset.')

            # if has saved, just load
            if os.path.exists(os.path.join(self.root, f'{self.name}.npy')):
                data = np.load(os.path.join(self.root, f'{self.name}.npy'), allow_pickle=True).item()
                self.imgs = data['imgs']
                self.targets = data['targets']
            else:
                for index in tqdm(range(len(self.imgs))):
                    impath = self.imgs[index][0]
                    if self.root is not None:
                        impath = self.root / impath
                    img = self.loader(impath)

                    self.imgs[index] = (img, *self.imgs[index][1:])

                # save self.imgs and targets to root
                data = {'imgs': self.imgs, 'targets': self.targets}
                np.save(os.path.join(self.root, f'{self.name}.npy'), data)

            print(f'[{datetime.now().strftime("%Y/%m/%d %H:%M:%S")}] DONE.')

        def __getitem__(self, index, return_concepts=False):
            """
            Returns next element in the dataset given the current index.

            :param index: index of the data to get.
            :return: loaded item.
            """

            img_description = self.imgs[index]
            impath = img_description[0]
            target = img_description[1]
            bbox = None
            concepts, position = None, None
            if len(img_description) == 3:  # with bbox
                bbox = img_description[2]
            elif len(img_description) == 4:
                concepts = img_description[2]
                position = img_description[3]

            if self.loaded:
                img = impath
            else:
                if self.root is not None:
                    impath = self.root / impath
                img = self.loader(impath)

            # If a bounding box is provided, crop the image before passing it to
            # any user-defined transformation.
            if bbox is not None:
                if isinstance(bbox, torch.Tensor):
                    bbox = bbox.tolist()
                img = crop(img, *bbox)

            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

            # If provide concepts and position,
            if return_concepts:
                return img, target, index, concepts, position
            else:
                return img, target, index

        def __len__(self):
            """
            Returns the total number of elements in the dataset.

            :return: Total number of dataset items.
            """

            return len(self.imgs)


class COBJ(CGQA):
    # def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
    #     super().__init__(root, train, transform, target_transform, download)

    def download_dataset(self):
        pass

    def _get_datasets(self, dataset_root, image_size=(128, 128), shuffle=False, seed=None,
                      mode='continual', num_samples_each_label=None, label_offset=0, load_set=None):

        """
        Create COBJ dataset, with given json files,
        containing instance tuples with shape (img_name, label).

        You may need to specify label_offset if relative label do not start from 0.

        :param dataset_root: Path to the dataset root folder.
        :param image_size: size of image.
        :param shuffle: If true, the train sample order (in json)
            in the incremental experiences is
            randomly shuffled. Default to False.
        :param seed: A valid int used to initialize the random number generator.
            Can be None.
        :param mode: Option [continual, sys, pro, sub, non, noc, nons, syss].
        :param num_samples_each_label: If specify a certain number of samples for each label,
            random sampling (build-in seed:1234,
            and replace=True if num_samples_each_label > num_samples, else False)
            is used to sample.
            Only for continual mode, only apply to train dataset.
        :param label_offset: specified if relative label not start from 0.
        :param load_set: train -> only pre-load train set;
            val -> only pre-load val set;
            test -> only pre-load test set.
            default None.

        :return data_sets defined by json file and label information.
        """
        img_folder_path = os.path.join(dataset_root, "CFST", "COBJ", "annotations")

        def preprocess_concept_to_integer(img_info, mapping_tuple_label_to_int_concepts):
            for item in img_info:
                item['concepts'] = [mapping_tuple_label_to_int_concepts[concept] for concept in item['label']]

        def preprocess_label_to_integer(img_info, mapping_tuple_label_to_int, prefix=''):
            for item in img_info:
                item['image'] = f"{prefix}{item['imageId']}.jpg"
                item['label'] = mapping_tuple_label_to_int[tuple(sorted(item['label']))]

        def formulate_img_tuples(images):
            """generate train_list and test_list: list with img tuple (path, label)"""
            img_tuples = []
            for item in images:
                instance_tuple = (item['image'], item['label'], item['concepts'])  # , item['boundingBox']
                img_tuples.append(instance_tuple)
            return img_tuples

        if mode == 'continual':
            train_json_path = os.path.join(img_folder_path, "O365_continual_train_crop.json")
            val_json_path = os.path.join(img_folder_path, "O365_continual_val_crop.json")
            test_json_path = os.path.join(img_folder_path, "O365_continual_test_crop.json")

            with open(train_json_path, 'r') as f:
                train_img_info = json.load(f)
            with open(val_json_path, 'r') as f:
                val_img_info = json.load(f)
            with open(test_json_path, 'r') as f:
                test_img_info = json.load(f)
            # img_info:
            # [{'newImageName': 'continual/val/59767',
            #   'comb': ['hat', 'leaves'],
            #   'objects': [{'imageName': '2416370', 'objName': 'hat',
            #                'attributes': ['red'], 'boundingBox': [52, 289, 34, 45]},...]
            #   'position': [4, 1]},...]

            '''preprocess labels to integers'''
            label_set = sorted(list(set([tuple(sorted(item['label'])) for item in val_img_info])))
            # [('building', 'sign'), ...]
            map_tuple_label_to_int = dict((item, idx + label_offset) for idx, item in enumerate(label_set))
            # {('building', 'sign'): 0, ('building', 'sky'): 1, ...}
            map_int_label_to_tuple = dict((idx + label_offset, item) for idx, item in enumerate(label_set))
            # {0: ('building', 'sign'), 1: ('building', 'sky'),...}
            '''preprocess concepts to integers'''
            concept_set = sorted(list(set([concept for item in val_img_info for concept in item['label']])))
            mapping_tuple_label_to_int_concepts = dict((item, idx) for idx, item in enumerate(concept_set))
            map_int_concepts_label_to_str = dict((idx, item) for idx, item in enumerate(concept_set))

            preprocess_concept_to_integer(train_img_info, mapping_tuple_label_to_int_concepts)
            preprocess_concept_to_integer(val_img_info, mapping_tuple_label_to_int_concepts)
            preprocess_concept_to_integer(test_img_info, mapping_tuple_label_to_int_concepts)

            preprocess_label_to_integer(train_img_info, map_tuple_label_to_int, prefix='continual/train/')
            preprocess_label_to_integer(val_img_info, map_tuple_label_to_int, prefix='continual/val/')
            preprocess_label_to_integer(test_img_info, map_tuple_label_to_int, prefix='continual/test/')

            '''if num_samples_each_label provided, sample images to balance each class for train set'''
            selected_train_images = []
            if num_samples_each_label is not None and num_samples_each_label > 0:
                imgs_each_label = dict()
                for item in train_img_info:
                    label = item['label']
                    if label in imgs_each_label:
                        imgs_each_label[label].append(item)
                    else:
                        imgs_each_label[label] = [item]
                build_in_seed = 1234
                build_in_rng = np.random.RandomState(seed=build_in_seed)
                for label, imgs in imgs_each_label.items():
                    selected_idxs = build_in_rng.choice(
                        np.arange(len(imgs)), num_samples_each_label,
                        replace=True if num_samples_each_label > len(imgs) else False)
                    for idx in selected_idxs:
                        selected_train_images.append(imgs[idx])
            else:
                selected_train_images = train_img_info

            '''generate train_list and test_list: list with img tuple (path, label)'''
            train_list = formulate_img_tuples(selected_train_images)
            val_list = formulate_img_tuples(val_img_info)
            test_list = formulate_img_tuples(test_img_info)
            # [('continual/val/59767.jpg', 0),...

            '''shuffle the train set'''
            if shuffle:
                rng = np.random.RandomState(seed=seed)
                order = np.arange(len(train_list))
                rng.shuffle(order)
                train_list = [train_list[idx] for idx in order]

            '''generate train_set and test_set using PathsDataset'''
            train_set = self.PathsDataset(
                root=img_folder_path,
                files=train_list,
                transform=transforms.Compose([transforms.Resize(image_size)]),
                loaded=load_set == 'train',
                name='con_train',
            )
            val_set = self.PathsDataset(
                root=img_folder_path,
                files=val_list,
                transform=transforms.Compose([transforms.Resize(image_size)]),
                loaded=load_set == 'val',
                name='con_val',
            )
            test_set = self.PathsDataset(
                root=img_folder_path,
                files=test_list,
                transform=transforms.Compose([transforms.Resize(image_size)]),
                loaded=load_set == 'test',
                name='con_test',
            )

            datasets = {'train': train_set, 'val': val_set, 'test': test_set}
            meta_info = {
                "concept_set": concept_set,
                "mapping_tuple_label_to_int_concepts": mapping_tuple_label_to_int_concepts,
                "map_int_concepts_label_to_str": map_int_concepts_label_to_str,
                "train_list": train_list, "val_list": val_list, "test_list": test_list}
            label_info = (label_set, map_tuple_label_to_int, map_int_label_to_tuple, meta_info)

        elif mode in ['sys', 'pro', 'non', 'noc']:  # no sub
            json_name = {'sys': 'O365_sys_fewshot_crop.json', 'pro': 'O365_pro_fewshot_crop.json',
                         'non': 'O365_non_fewshot_crop.json', 'noc': 'O365_noc_fewshot_crop.json'}[mode]
            json_path = os.path.join(img_folder_path, json_name)
            with open(json_path, 'r') as f:
                img_info = json.load(f)
            label_set = sorted(list(set([tuple(sorted(item['label'])) for item in img_info])))
            map_tuple_label_to_int = dict((item, idx + label_offset) for idx, item in enumerate(label_set))
            map_int_label_to_tuple = dict((idx + label_offset, item) for idx, item in enumerate(label_set))
            preprocess_label_to_integer(img_info, map_tuple_label_to_int, prefix=f'fewshot/{mode}/')
            img_list = formulate_img_tuples(img_info)
            dataset = self.PathsDataset(
                root=img_folder_path,
                files=img_list,
                transform=transforms.Compose([transforms.Resize(image_size)]),
                loaded=True,
                name=f'few_{mode}',
            )

            datasets = {'dataset': dataset}
            label_info = (label_set, map_tuple_label_to_int, map_int_label_to_tuple)

        else:
            raise Exception(f'Un-implemented mode "{mode}".')

        return datasets, label_info


class MNIST_RGB(datasets.MNIST):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(MNIST_RGB, self).__init__(root, transform=transform, target_transform=target_transform, download=download)
        self.train = train  # training set or test set

        if self._check_legacy_exist():
            self.data, self.targets = self._load_legacy_data()
            return

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data, self.targets = self._load_data()

    def _check_legacy_exist(self):
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False

        return all(
            check_integrity(os.path.join(self.processed_folder, file)) for file in (self.training_file, self.test_file)
        )

    def _load_legacy_data(self):
        # This is for BC only. We no longer cache the data in a custom binary, but simply read from the raw data
        # directly.
        data_file = self.training_file if self.train else self.test_file
        return torch.load(os.path.join(self.processed_folder, data_file))

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.raw_folder, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.raw_folder, label_file))

        return data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        try:
            img = Image.fromarray(img.numpy(), mode='L').convert('RGB')
        except:
            pass

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class FashionMNIST(MNIST_RGB):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``FashionMNIST/raw/train-images-idx3-ubyte``
            and  ``FashionMNIST/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    mirrors = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"]

    resources = [
        ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
        ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
        ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
    ]
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

class NotMNIST(MNIST_RGB):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        self.url = 'https://github.com/facebookresearch/Adversarial-Continual-Learning/raw/main/data/notMNIST.zip'
        self.filename = 'notMNIST.zip'

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                download_url(self.url, root, filename=self.filename)

            import zipfile
            zip_ref = zipfile.ZipFile(fpath, 'r')
            zip_ref.extractall(root)
            zip_ref.close()

        if self.train:
            fpath = os.path.join(root, 'notMNIST', 'Train')

        else:
            fpath = os.path.join(root, 'notMNIST', 'Test')


        X, Y = [], []
        folders = os.listdir(fpath)

        for folder in folders:
            folder_path = os.path.join(fpath, folder)
            for ims in os.listdir(folder_path):
                try:
                    img_path = os.path.join(folder_path, ims)
                    X.append(np.array(Image.open(img_path).convert('RGB')))
                    Y.append(ord(folder) - 65)  # Folders are A-J so labels will be 0-9
                except:
                    print("File {}/{} is broken".format(folder, ims))
        self.data = np.array(X)
        self.targets = Y

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        try:
            img = Image.fromarray(img)
        except:
            pass

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class SVHN(datasets.SVHN):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(SVHN, self).__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat["X"]
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.targets = loaded_mat["y"].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.targets, self.targets == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))
        self.classes = np.unique(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self) -> None:
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)


class Flowers102(datasets.Flowers102):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(Flowers102, self).__init__(root, transform=transform, target_transform=target_transform, download=download)
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._base_folder = Path(self.root) / "flowers-102"
        self._images_folder = self._base_folder / "jpg"

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        from scipy.io import loadmat

        set_ids = loadmat(self._base_folder / self._file_dict["setid"][0], squeeze_me=True)
        image_ids = set_ids[self._splits_map[self._split]].tolist()

        labels = loadmat(self._base_folder / self._file_dict["label"][0], squeeze_me=True)
        image_id_to_label = dict(enumerate(labels["labels"].tolist(), 1))

        self.targets = []
        self._image_files = []
        for image_id in image_ids:
            self.targets.append(image_id_to_label[image_id] - 1) # -1 for 0-based indexing
            self._image_files.append(self._images_folder / f"image_{image_id:05d}.jpg")
        self.classes = list(set(self.targets))
    
    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self.targets[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_integrity(self):
        if not (self._images_folder.exists() and self._images_folder.is_dir()):
            return False

        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            if not check_integrity(str(self._base_folder / filename), md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            return
        download_and_extract_archive(
            f"{self._download_url_prefix}{self._file_dict['image'][0]}",
            str(self._base_folder),
            md5=self._file_dict["image"][1],
        )
        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            download_url(self._download_url_prefix + filename, str(self._base_folder), md5=md5)

class StanfordCars(datasets.StanfordCars):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        try:
            import scipy.io as sio
        except ImportError:
            raise RuntimeError("Scipy is not found. This dataset needs to have scipy installed: pip install scipy")

        super(StanfordCars, self).__init__(root, transform=transform, target_transform=target_transform, download=download)

        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = pathlib.Path(root) / "stanford_cars"
        devkit = self._base_folder / "devkit"

        if self._split == "train":
            self._annotations_mat_path = devkit / "cars_train_annos.mat"
            self._images_base_path = self._base_folder / "cars_train"
        else:
            self._annotations_mat_path = self._base_folder / "cars_test_annos_withlabels.mat"
            self._images_base_path = self._base_folder / "cars_test"

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._samples = [
            (
                str(self._images_base_path / annotation["fname"]),
                annotation["class"] - 1,  # Original target mapping  starts from 1, hence -1
            )
            for annotation in sio.loadmat(self._annotations_mat_path, squeeze_me=True)["annotations"]
        ]

        self.classes = sio.loadmat(str(devkit / "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Returns pil_image and class_id for given index"""
        image_path, target = self._samples[idx]
        pil_image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return pil_image, target

    def download(self) -> None:
        if self._check_exists():
            return

        download_and_extract_archive(
            url="https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz",
            download_root=str(self._base_folder),
            md5="c3b158d763b6e2245038c8ad08e45376",
        )
        if self._split == "train":
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_train.tgz",
                download_root=str(self._base_folder),
                md5="065e5b463ae28d29e77c1b4b166cfe61",
            )
        else:
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_test.tgz",
                download_root=str(self._base_folder),
                md5="4ce7ebf6a94d07f1952d94dd34c4d501",
            )
            download_url(
                url="https://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat",
                root=str(self._base_folder),
                md5="b0a2b23655a3edd16d84508592a98d10",
            )

    def _check_exists(self) -> bool:
        if not (self._base_folder / "devkit").is_dir():
            return False

        return self._annotations_mat_path.exists() and self._images_base_path.is_dir()

class CUB200(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        self.url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'
        self.filename = 'CUB_200_2011.tgz'

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                download_url(self.url, root, filename=self.filename)

        if not os.path.exists(os.path.join(root, 'CUB_200_2011')):
            import zipfile
            zip_ref = zipfile.ZipFile(fpath, 'r')
            zip_ref.extractall(root)
            zip_ref.close()

            import tarfile
            tar_ref = tarfile.open(os.path.join(root, 'CUB_200_2011.tgz'), 'r')
            tar_ref.extractall(root)
            tar_ref.close()

            self.split()
        
        if self.train:
            fpath = os.path.join(root, 'CUB_200_2011', 'train')

        else:
            fpath = os.path.join(root, 'CUB_200_2011', 'test')

        self.data = datasets.ImageFolder(fpath, transform=transform)

    def split(self):
        train_folder = self.root + 'CUB_200_2011/train'
        test_folder = self.root + 'CUB_200_2011/test'

        if os.path.exists(train_folder):
            rmtree(train_folder)
        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(train_folder)
        os.mkdir(test_folder)

        images = self.root + 'CUB_200_2011/images.txt'
        train_test_split = self.root + 'CUB_200_2011/train_test_split.txt'

        with open(images, 'r') as image:
            image_paths = image.readlines()
            with open(train_test_split, 'r') as f:
                i = 0
                for line in f:
                    image_path = image_paths[i]
                    image_path = image_path.replace('\n', '').split(' ')[-1]
                    class_name = image_path.split('/')[0]
                    src = self.root + 'CUB_200_2011/images/' + class_name

                    if line.split(' ')[-1].replace('\n', '') == '1':
                        if not os.path.exists(train_folder + '/' + class_name):
                            os.mkdir(train_folder + '/' + class_name)
                        dst = train_folder + '/' + image_path
                    else:
                        if not os.path.exists(test_folder + '/' + class_name):
                            os.mkdir(test_folder + '/' + class_name)
                        dst = test_folder + '/' + image_path
                    
                    move(src, dst)
                    i += 1

class TinyImagenet(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        self.url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        self.filename = 'tiny-imagenet-200.zip'

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                download_url(self.url, root, filename=self.filename)
        
        if not os.path.exists(os.path.join(root, 'tiny-imagenet-200')):
            import zipfile
            zip_ref = zipfile.ZipFile(fpath, 'r')
            zip_ref.extractall(os.path.join(root))
            zip_ref.close()

            self.split()

        if self.train:
            fpath = root + 'tiny-imagenet-200/train'

        else:
            fpath = root + 'tiny-imagenet-200/test'
        
        self.data = datasets.ImageFolder(fpath, transform=transform)

    def split(self):
        test_folder = self.root + 'tiny-imagenet-200/test'

        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(test_folder)

        val_dict = {}
        with open(self.root + 'tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
            for line in f.readlines():
                split_line = line.split('\t')
                val_dict[split_line[0]] = split_line[1]
                
        paths = glob.glob(self.root + 'tiny-imagenet-200/val/images/*')
        for path in paths:
            if '\\' in path:
                path = path.replace('\\', '/')
            file = path.split('/')[-1]
            folder = val_dict[file]
            if not os.path.exists(test_folder + '/' + folder):
                os.mkdir(test_folder + '/' + folder)
                os.mkdir(test_folder + '/' + folder + '/images')
            
            
        for path in paths:
            if '\\' in path:
                path = path.replace('\\', '/')
            file = path.split('/')[-1]
            folder = val_dict[file]
            src = path
            dst = test_folder + '/' + folder + '/images/' + file
            move(src, dst)
        
        rmtree(self.root + 'tiny-imagenet-200/val')

class Scene67(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        image_url = 'http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar'
        train_annos_url = 'http://web.mit.edu/torralba/www/TrainImages.txt'
        test_annos_url = 'http://web.mit.edu/torralba/www/TestImages.txt'
        urls = [image_url, train_annos_url, test_annos_url]
        image_fname = 'indoorCVPR_09.tar'
        self.train_annos_fname = 'TrainImage.txt'
        self.test_annos_fname = 'TestImage.txt'
        fnames = [image_fname, self.train_annos_fname, self.test_annos_fname]

        for url, fname in zip(urls, fnames):
            fpath = os.path.join(root, fname)
            if not os.path.isfile(fpath):
                if not download:
                    raise RuntimeError('Dataset not found. You can use download=True to download it')
                else:
                    print('Downloading from ' + url)
                    download_url(url, root, filename=fname)
        if not os.path.exists(os.path.join(root, 'Scene67')):
            import tarfile
            with tarfile.open(os.path.join(root, image_fname)) as tar:
                tar.extractall(os.path.join(root, 'Scene67'))

            self.split()

        if self.train:
            fpath = os.path.join(root, 'Scene67', 'train')

        else:
            fpath = os.path.join(root, 'Scene67', 'test')

        self.data = datasets.ImageFolder(fpath, transform=transform)

    def split(self):
        if not os.path.exists(os.path.join(self.root, 'Scene67', 'train')):
            os.mkdir(os.path.join(self.root, 'Scene67', 'train'))
        if not os.path.exists(os.path.join(self.root, 'Scene67', 'test')):
            os.mkdir(os.path.join(self.root, 'Scene67', 'test'))
        
        train_annos_file = os.path.join(self.root, self.train_annos_fname)
        test_annos_file = os.path.join(self.root, self.test_annos_fname)

        with open(train_annos_file, 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                src = self.root + 'Scene67/' + 'Images/' + line
                dst = self.root + 'Scene67/' + 'train/' + line
                if not os.path.exists(os.path.join(self.root, 'Scene67', 'train', line.split('/')[0])):
                   os.mkdir(os.path.join(self.root, 'Scene67', 'train', line.split('/')[0]))
                move(src, dst)
        
        with open(test_annos_file, 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                src = self.root + 'Scene67/' + 'Images/' + line
                dst = self.root + 'Scene67/' + 'test/' + line
                if not os.path.exists(os.path.join(self.root, 'Scene67', 'test', line.split('/')[0])):
                   os.mkdir(os.path.join(self.root, 'Scene67', 'test', line.split('/')[0]))
                move(src, dst)


class Imagenet_R(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        self.url = 'https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar'
        self.filename = 'imagenet-r.tar'

        self.fpath = os.path.join(root, 'imagenet-r')
        if not os.path.isfile(self.fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                download_url(self.url, root, filename=self.filename)

        if not os.path.exists(os.path.join(root, 'imagenet-r')):
            import tarfile
            tar_ref = tarfile.open(os.path.join(root, self.filename), 'r')
            tar_ref.extractall(root)
            tar_ref.close()
        
        if not os.path.exists(self.fpath + '/train') and not os.path.exists(self.fpath + '/test'):
            self.dataset = datasets.ImageFolder(self.fpath, transform=transform)
            
            train_size = int(0.8 * len(self.dataset))
            val_size = len(self.dataset) - train_size
            
            train, val = torch.utils.data.random_split(self.dataset, [train_size, val_size])
            train_idx, val_idx = train.indices, val.indices
    
            self.train_file_list = [self.dataset.imgs[i][0] for i in train_idx]
            self.test_file_list = [self.dataset.imgs[i][0] for i in val_idx]

            self.split()
        
        if self.train:
            fpath = self.fpath + '/train'

        else:
            fpath = self.fpath + '/test'

        self.data = datasets.ImageFolder(fpath, transform=transform)

    def split(self):
        train_folder = self.fpath + '/train'
        test_folder = self.fpath + '/test'

        if os.path.exists(train_folder):
            rmtree(train_folder)
        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(train_folder)
        os.mkdir(test_folder)

        for c in self.dataset.classes:
            if not os.path.exists(os.path.join(train_folder, c)):
                os.mkdir(os.path.join(os.path.join(train_folder, c)))
            if not os.path.exists(os.path.join(test_folder, c)):
                os.mkdir(os.path.join(os.path.join(test_folder, c)))
        
        for path in self.train_file_list:
            if '\\' in path:
                path = path.replace('\\', '/')
            src = path
            dst = os.path.join(train_folder, '/'.join(path.split('/')[-2:]))
            move(src, dst)

        for path in self.test_file_list:
            if '\\' in path:
                path = path.replace('\\', '/')
            src = path
            dst = os.path.join(test_folder, '/'.join(path.split('/')[-2:]))
            move(src, dst)
        
        for c in self.dataset.classes:
            path = os.path.join(self.fpath, c)
            rmtree(path)


class DomainNet(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train
        self.domain_names = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch", ]
        
        if self.train:
            image_list_paths = [os.path.join(self.root, d + '_train.txt') for d in self.domain_names]
            imgs = []
            for i in range(len(self.domain_names)):
                img_path = image_list_paths[i]
                image_list = open(img_path).readlines()
                imgs += [(val.split()[0], int(val.split()[1]) + i * 345) for val in image_list]
            train_x = []
            train_y = []
            for item in imgs:
                train_x.append(os.path.join(self.root, item[0]))
                train_y.append(item[1])
            self.data = np.array(train_x)
            self.targets = np.array(train_y)
        else:
            image_list_paths = [os.path.join(self.root, d + '_test.txt') for d in self.domain_names]
            imgs = []
            for i in range(len(self.domain_names)):
                img_path = image_list_paths[i]
                image_list = open(img_path).readlines()
                imgs += [(val.split()[0], int(val.split()[1]) + i * 345) for val in image_list]
            test_x = []
            test_y = []
            for item in imgs:
                test_x.append(os.path.join(self.root, item[0]))
                test_y.append(item[1])
            self.data = np.array(test_x)
            self.targets = np.array(test_y)
        self.classes = [i for i in range(2070)]

    def __getitem__(self, idx):
        with open(self.data[idx], 'rb') as f:
            image = Image.open(f).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
            
        target = self.targets[idx]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target
    
    def __len__(self):
        return len(self.targets)
    

# class Imagenet_R(torch.utils.data.Dataset):
#     def __init__(self, root, train=True, transform=None, target_transform=None, download=False):        
#         self.root = os.path.expanduser(root)
#         self.transform = transform
#         self.target_transform=target_transform
#         self.train = train

#         self.url = 'https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar'
#         self.filename = 'imagenet-r.tar'

#         self.fpath = os.path.join(root, 'imagenet-r')
#         self.pre_data_path = '/'.join(root.split('/')[:-1])

#         if not os.path.isfile(self.fpath):
#             if not download:
#                raise RuntimeError('Dataset not found. You can use download=True to download it')
#             else:
#                 print('Downloading from '+self.url)
#                 download_url(self.url, root, filename=self.filename)

#         if not os.path.exists(os.path.join(root, 'imagenet-r')):
#             import tarfile
#             tar_ref = tarfile.open(os.path.join(root, self.filename), 'r')
#             tar_ref.extractall(root)
#             tar_ref.close()

#         import yaml
#         with open(os.path.join(self.root, 'imagenet-r_train.yaml'), 'r') as f:
#             train_file = yaml.safe_load(f)
#         with open(os.path.join(self.root, 'imagenet-r_test.yaml'), 'r') as f:
#             test_file = yaml.safe_load(f)
#         assert len(train_file['data']) == len(train_file['targets'])
#         assert len(test_file["data"]) == len(test_file["targets"])

#         if self.train:
#             self.data = np.array(train_file["data"])
#             self.targets = np.array(train_file["targets"])
#         else:
#             self.data = np.array(test_file["data"])
#             self.targets = np.array(test_file["targets"])
    
#         self.classes = np.unique(self.targets)

#     def __getitem__(self, idx):
#         with open(os.path.join(self.pre_data_path, self.data[idx]), 'rb') as f:
#             image = Image.open(f).convert('RGB')
#         if self.transform is not None:
#             image = self.transform(image)
            
#         target = self.targets[idx]
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#         return image, target
    
#     def __len__(self):
#         return len(self.targets)

        
