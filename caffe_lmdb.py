import torch
import lmdb
import caffe2
from caffe2.proto import caffe2_pb2
from PIL import Image
import io

import numpy as np

from joblib import Parallel, delayed


class CaffeLMDB:
    def __init__(self, root, rank=0, world_size=1, transform=None, target_transform=None, max_readers=1, cache=False, label_type="int"):
        # https://github.com/pytorch/vision/blob/master/torchvision/datasets/lsun.py
        self.root = root
        
        env = lmdb.open(root,readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)
        self.nb = env.stat()["entries"]
        env.close()

        self.transform = transform
        self.target_transform = target_transform
        # self.shuffled_data = np.arange(self.nb)
        # self.sampler = torch.utils.data.distributed.DistributedSampler(self, num_replicas=world_size, rank=rank)
        self.cache = cache
        self.cache_dict = {}
        self.label_type = label_type
        self.env = None
    
    def __getitem__(self, i):
        if self.env is None:
            self.env = lmdb.open(self.root,readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)
        if self.cache:
            if i in self.cache_dict:
                img, target = self.cache_dict[i]
                img, target = self.get(i)
                self.cache_dict[i] = img, target
        else:
            img, target = self.get(i)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def get(self, i):
        with self.env.begin(write=False) as txn:
            key = f"{i}"
            value = txn.get(key.encode("ascii"))
        try:
            image_data, label_data = caffe2_pb2.TensorProtos.FromString(value).protos
        except Exception:
            random_index = np.random.randint(0, self.nb)
            return self.get(random_index)
        imgbuf = image_data.string_data[0]
        if self.label_type == "int":
            target = (label_data.int32_data[0])
        elif self.label_type  == "str":
            target = (label_data.string_data[0])
        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        try:
            img = Image.open(buf).convert("RGB")
        except Exception:
            random_index = np.random.randint(0, self.nb)
            return self.get(random_index)
        return img, target

    def __len__(self):
        return (self.nb)

def caffe_lmdb_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.env = lmdb.open(dataset.root,readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)

def caffe_lmdb_multiple_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    nb = worker_info.num_workers
    dataset = worker_info.dataset  # the dataset copy in this worker process
    for ds in dataset.datasets:
        ds.env = lmdb.open(ds.root,readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)

class CaffeLMDBMultiple:

    def __init__(self, paths, transform=None, target_transform=None, label_type="int"):
        self.datasets = [CaffeLMDB(path, transform=transform, target_transform=target_transform, label_type=label_type) for path in paths]
        self.nb = sum(map(len, self.datasets))
        start = 0
        self.dataset_inds = []
        for ds in self.datasets:
            end = start + len(ds)
            self.dataset_inds.append((start, end))
            start = end
        self._transform = transform
        self._target_transform = target_transform
    
    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, new_transform):
        for ds in self.datasets:
            ds.transform = new_transform
        self._transform = new_transform

    def __getitem__(self, idx):
        dataset_idx = None
        for i, (start, end) in enumerate(self.dataset_inds):
            if idx >= start and idx < end:
                dataset_idx = i
                break
        # raise ValueError(f"DATASET {dataset_idx}, {idx}, {self.nb}")
        dataset = self.datasets[dataset_idx]
        start, end = self.dataset_inds[dataset_idx]
        offset = idx - start
        return dataset[offset]

    def __len__(self):
        return self.nb
