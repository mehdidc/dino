import ast
import json
import logging
import math
import os
import random
from dataclasses import dataclass

import braceexpand
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler


try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler


def preprocess_txt(text):
    return tokenize([str(text)])[0]


def get_dataset_size(shards):
    if type(shards) == list:
        return None, len(shards)

    shards_list = list(braceexpand.braceexpand(shards))
    dir_path = os.path.dirname(shards)
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # cc3m-train: 2905954
        # cc12m: 10968539
        # LAION-400m: 407332084
    num_shards = len(shards_list)
    return total_size, num_shards

def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption(sample):
    return 'txt' in sample


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


def get_wds_dataset(args, preprocess_img, preprocess_target, is_train, grouping=False):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None

    num_samples, num_shards = get_dataset_size(input_shards)
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    'Currently, number of dataset samples must be specified for training dataset. '
                    'Please specify via `--train-num-samples` if no dataset length info present.')
        else:
            num_samples = args.val_num_samples or 0  # eval will just exhaust the iterator if not specified
     
    pipeline = [wds.SimpleShardList(input_shards)]
    # at this point we have an iterator over all the shards
    if is_train:
        pipeline.extend([
            wds.detshuffle(bufsize=_SHARD_SHUFFLE_SIZE, initial=_SHARD_SHUFFLE_INITIAL, seed=args.seed),
            wds.split_by_node,
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker at each node
            wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
                rng=random.Random(args.seed)),
            #wds.repeatedly,  # FIXME determine if this is beneficial
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    
    if args.label_type == "str":
        pipeline.append(wds.select(filter_no_caption))

    pipeline.extend([
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image=args.input_col, text=args.output_col),
        wds.map_dict(image=preprocess_img, text=preprocess_target),
    ])
    if grouping:
        def group(data):
            for sample in data:
                yield tuple(sample["image"]) + tuple([sample["text"]])
        pipeline.append(wds.then(group))
    pipeline.extend([
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train),
    ])

    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        # roll over and repeat a few samples to get same number of full batches on each node
        global_batch_size = args.batch_size * args.world_size
        num_batches = (num_samples // global_batch_size)
        print("WK", args.batch_size, args.world_size, num_batches)
        num_workers = max(1, args.workers)
        num_worker_batches = (num_batches // num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers + 1
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        # FIXME detshuffle uses same seed each epoch unless workers are persistent
        # this seems like a WDS bug, currently waiting for clarification
        persistent_workers=True,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader, None)
