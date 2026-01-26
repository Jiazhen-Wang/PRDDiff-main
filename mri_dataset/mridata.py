from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pickle, os
import torch
import random
import torchvision
import scipy.io as scio
import fastmri


def load_data(
        *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),

    )
    if deterministic:
        # print('one')
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        # print("one")
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


# 测试数据集
def load_data_test(
        *, data_dir, batch_size, image_size, class_cond=False, deterministic=True
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),

    )

    if deterministic:
        print('one')
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        print("one")
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        return loader


# 获取一个路径下的所有文件
def _list_image_files_recursively(data_dir):
    results = []
    # 暂时猜测listdir是输出data_dir下的所有自文件夹名字 输出形式为列表
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "mat", "pt"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        """
        fastMRI is preprocessed and stored as pickle files, where kspace raw data is stored under 'img';
        """
        path = self.local_images[idx]
        out = scio.loadmat(path)['img']
        out_mo = np.max(np.abs(out[0] + out[1] * 1j))
        out = out / out_mo
        return out, {}


class MRIDataset(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.data_path = opt["dataroot"]
        self.HR_size = opt["full_size"]
        self.local_images = _list_image_files_recursively(self.data_path)[0:][::1]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        """
        fastMRI is preprocessed and stored as pickle files, where kspace raw data is stored under 'img';
        """
        path = self.local_images[idx]
        ii = path.split('/')
        id = ii[-1]
        out = scio.loadmat(path)['img']
        out_mo = np.max(np.abs(out[0] + out[1] * 1j))
        out = out / out_mo

        return out, id


class MRIDatasetID(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.data_path = opt["dataroot"]
        self.HR_size = opt["full_size"]
        self.local_images = _list_image_files_recursively(self.data_path)[0:][::1]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        """
        fastMRI is preprocessed and stored as pickle files, where kspace raw data is stored under 'img';
        """
        path = self.local_images[idx]
        ii = path.split('/')
        id = ii[-1]
        out = scio.loadmat(path)['img']
        out_mo = np.max(np.abs(out[0] + out[1] * 1j))
        out = out / out_mo
        out = np.float32(out)
        return out, id






