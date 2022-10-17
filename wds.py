import os
import sys
import math
import logging
import functools
import braceexpand
import random
import pdb
import json
from glob import glob

import pandas as pd
import numpy as np
import pyarrow as pa
from PIL import Image

from typing import Union
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, get_worker_info
from torch.utils.data.distributed import DistributedSampler
import torchvision.datasets as datasets
from webdataset.utils import identity
import webdataset as wds
from webdataset.shardlists import IterableDataset, Composable, ShardSample, SimpleShardSample

from distributed import world_info_from_env

def tokenizer(x):
    return x

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t"):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = tokenize([str(self.captions[idx])])[0]
        return images, texts


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler


def preprocess_txt(text):
    return tokenizer(str(text))


def get_dataset_size(args, shards):
    shards_list = list(braceexpand.braceexpand(shards))
    dir_path = os.path.dirname(shards)
    if 'sizes.json' in os.listdir(dir_path):
        sizes_filename = os.path.join(dir_path, 'sizes.json')
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum(
            [int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif args.num_samples_per_epoch:
        total_size = args.num_samples_per_epoch
    elif '__len__' in os.listdir(dir_path):
        total_size = eval(open(os.path.join(dir_path, '__len__'), 'r').read())
    else:
        # name / path based hacks (NOTE: specific to a given download / instances of dataset in terms of samples)
        if 'cc3m-train' in shards:
            total_size = 2905954
        elif 'imagenet-1K' in shards:
            total_size = 1281167
        elif 'cc12m' in shards:
            total_size = 10968539
        elif 'laion' in dir_path.lower():
            total_size = 407332084
        else:
            raise ValueError(f'Could not find dataset size in {dir_path}')
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path  = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader, sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


class DistPytorchEnv:
    """A class encapsulating the PyTorch node/worker environment."""

    def __init__(self, group=None):
        """Initialize rank/worker information."""
        import socket

        super().__init__()
        self.rank = None
        self.worker = None
        self.group = group
        self.nodeinfo = (socket.gethostname(), os.getpid())
        self.update_env()

    def update_env(self):
        """Update information about node and worker environment.
        This code is written this way because the torch.distributed info is
        available only in the environment where the loader is created.
        This class retains that environment info when it is serialized.
        """
        from webdataset import gopen

        try:
            import torch
            import torch.distributed
        except Exception:
            return

        if self.rank is None:
            if hvd is not None and hvd.is_initialized():
                self.rank = hvd.rank(), hvd.size()
            elif torch.distributed.is_available() and torch.distributed.is_initialized():
                group = self.group or torch.distributed.group.WORLD
                self.rank = torch.distributed.get_rank(group=group), \
                            torch.distributed.get_world_size(group=group)
            else:
                _, rank, world_size = world_info_from_env()
                if world_size > 1:
                    self.rank = (rank, world_size)

        if self.worker is None:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                self.worker = worker_info.id, worker_info.num_workers

        gopen.info["nodeinfo"] = self.nodeinfo
        gopen.info["rank"], gopen.info["size"] = self.rank or (-1, -1)
        gopen.info["worker_id"], gopen.info["num_workers"] = self.worker or (-1, -1)


class DistShardList(IterableDataset, DistPytorchEnv, Composable):
    """An iterable dataset yielding a list of urls.
    This understands the PyTorch distributed and worker APIs and splits shards
    accordingly.
    """

    def __init__(
        self,
        urls,
        epoch_shuffle=False,
        shuffle=True,
        split_by_worker=True,
        split_by_node=True,
        verbose=False,
    ):
        """Create a ShardList.
        :param urls: a list of URLs as a Python list or brace notation string
        :param shuffle: shuffle samples before iterating
        :param split_by_node: split shards by node if True
        :param split_by_worker: split shards by worker if True
        :param group: group used for determining rank/world_size
        If WDS_SHUFFLE is in the environment, it is used for shuffling shards prior
        to splitting; this assigns different shards to different nodes on each epoch.
        """
        super().__init__()

        self.verbose = verbose
        if self.verbose:
            print("PytorchShardList init")
        self.epoch = -1
        self.epoch_shuffle = epoch_shuffle
        self.shuffle = shuffle
        self.split_by_worker = split_by_worker
        self.split_by_node = split_by_node
        if not isinstance(urls, ShardSample):
            urls = SimpleShardSample(urls)
        self.shardsample = urls

    def set_epoch(self, epoch):
        """Set the current epoch. Used for per-node shuffling."""
        self.epoch = epoch - 1

    def __iter__(self):
        """Return an iterator over the shards."""
        self.epoch += 1
        if hasattr(self.shardsample, "set_epoch"):
            self.shardsample.set_epoch(self.epoch)
        self.update_env()
        urls = self.shardsample.sample()
        if self.epoch_shuffle:
            if "WDS_EPOCH" not in os.environ:
                raise ValueError(
                    "when specifying epoch_shuffle, you must provide the epoch in the WDS_EPOCH environment variable"
                )
            epoch = int(os.environ["WDS_EPOCH"])
            if self.verbose:
                print(f"PytorchShardList epochshuffle {epoch}")
            random.Random(epoch).shuffle(urls)
        if self.split_by_node:
            rank, world = self.rank or (0, 1)
            if self.verbose:
                print(f"PytorchShardList rank {rank} of {world}")
            urls = urls[rank::world]
        if self.split_by_worker:
            worker, nworkers = self.worker or (0, 1)
            if self.verbose:
                print(f"PytorchShardList worker {worker} of {nworkers}")
            urls = urls[worker::nworkers]
        if self.shuffle:
            random.Random(self.epoch + 17).shuffle(urls)
        if self.verbose:
            print(f"PytorchShardList got {len(urls)} urls")
        for url in urls:
            yield dict(
                url=url,
                __url__=url,
                __worker__=str(self.worker),
                __rank__=str(self.rank),
                __nodeinfo__=str(self.nodeinfo),
            )


def filter_no_caption(sample):
    return 'txt' in sample

def pytorch_worker_seed():
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour the seed already created for pytorch dataloader workers if it exists
        return worker_info.seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()

class ResampledShards2(IterableDataset, Composable,):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.
        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls = wds.shardlists.expand_urls(urls) if type(urls) == str else urls
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = pytorch_worker_seed if worker_seed is None else worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        #if isinstance(self.epoch, SharedEpoch):
        #    epoch = self.epoch.get_value()
        #else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
        self.epoch += 1
        epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic, worker seed should be deterministic due to arg.seed
            self.rng.seed(self.worker_seed() + epoch)
        for _ in range(self.nshards):
            url = self.rng.choice(self.urls)
            yield dict(url=url)


def get_wds_dataset(args, preprocess_img, is_train, num_batches=None, resampled=False):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    # The following code is adapted from https://github.com/tmbdev/webdataset-examples/blob/master/main-wds.py
    num_samples, num_shards = get_dataset_size(args, input_shards)
    if num_batches is None:
        if is_train and args.distributed:
            max_shards_per_node = math.ceil(num_shards / args.world_size)
            num_samples = args.world_size * (num_samples * max_shards_per_node // num_shards)
            num_batches = num_samples // (args.batch_size * args.world_size)
            num_samples = num_batches * args.batch_size * args.world_size
        else:
            num_batches = num_samples // args.batch_size
    if os.path.isdir(input_shards):
        input_shards = glob(os.path.join(input_shards, "**", "*.tar"))
    print(input_shards, resampled)
    if resampled:
        shardlist = ResampledShards2(input_shards, deterministic=False, nshards=num_batches)
    else:
        shardlist = DistShardList(
            input_shards,
            epoch_shuffle=is_train,
            split_by_node=is_train  
        )
    def group(data):
        for sample in data:
            yield tuple(sample["image"]) #+ tuple([sample["text"]])

    dataset = (
        wds.WebDataset(shardlist)
        .select(filter_no_caption)
        .decode("pil", handler=wds.ignore_and_continue)
        .rename(image="jpg;png")
        .map_dict(image=preprocess_img)
        .then(group)
        .batched(args.batch_size, partial=not is_train or not args.distributed)
    )
    dataloader = wds.WebLoader(
        dataset, batch_size=None, shuffle=False, num_workers=args.workers,
    )
    if not resampled:
        if is_train and args.distributed:
            # With DDP, we need to make sure that all nodes get the same number of batches;
            # we do that by reusing a little bit of data.
            dataloader = dataloader.repeat(2).slice(num_batches)
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader, None)


def get_csv_dataset(args, preprocess_fn, is_train):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extention {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data:
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data


from torchvision import datasets, transforms
import utils
class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, stack=False):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])
        self.stack = stack

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        if self.stack:
            return [torch.cat(crops[0:2], dim=0), torch.cat(crops[2:], dim=0)]
        else:
            return crops



if __name__ == "__main__":
    p = DataAugmentationDINO([0.4, 1], [0.4, 1], 10)
    class Args:
        distributed = False
        batch_size = 2
        train_data = "/p/scratch/ccstdl/katta1/LAION-400M/laion400m-dat-release/{00000..41455}.tar"
        workers = 1
    def preprocess_img(x):
        return x
        #return torch.ones(2,3,64,64)
        # return x, x
    is_train = True
    import os
    os.environ['WDS_EPOCH'] = '0'
    ds =  get_wds_dataset(Args, p, is_train)
    for x in ds.dataloader:
        print(x[0].shape, x[1].shape)
    print(ds.dataloader.num_samples)
