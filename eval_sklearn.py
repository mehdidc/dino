# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
import sys
import argparse
import numpy as np

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import fewshot

def extract_feature_pipeline(args, path):
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # dataset_train = ReturnIndexDataset(os.path.join(args.data_path, "train"), transform=transform)
    # dataset_val = ReturnIndexDataset(os.path.join(args.data_path, "val"), transform=transform)
    # dataset_train = ReturnIndexDataset(os.path.join(args.data_path, "train"), transform=transform)
    # dataset_val = ReturnIndexDataset(os.path.join(args.data_path, "val"), transform=transform)
    if args.dataset == "image_folder":
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=transform)
        dataset_val = datasets.ImageFolder(os.path.join(args.data_path, "val"), transform=transform)
    elif args.dataset == "caffe_lmdb":
        from caffe_lmdb import CaffeLMDB
        dataset_train = CaffeLMDB(os.path.join(args.data_path, "train"), transform=transform)
        dataset_train.classes = np.arange(args.nb_classes)
        dataset_val = CaffeLMDB(os.path.join(args.data_path, "val"), transform=transform)
        dataset_val.classes = np.arange(args.nb_classes)

    else:
        raise ValueError(args.dataset)

    if args.shots:
        print("Few shot", args.shots)
        indices = fewshot.find_fewshot_indices(dataset_train, args.shots)
        dataset_train = torch.utils.data.Subset(dataset_train, indices)
    sampler = None
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=0)
        model.fc = nn.Identity()
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    model.cuda()
    utils.load_pretrained_weights(model, path, args.checkpoint_key, args.arch, args.patch_size)
    # ckpt = torch.load(args.pretrained_weights, map_location='cpu')
    # weights = ckpt[args.checkpoint_key]
    # model.load_state_dict(weights)
    # model = model.encode_image_base
    model.eval()

    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features, train_labels = extract_features(model, data_loader_train, args.use_cuda, nb=args.nb_batches, n_last_blocks=args.n_last_blocks, avgpool_patchtokens=args.avgpool_patchtokens)
    print("Extracting features for val set...")
    test_features, test_labels = extract_features(model, data_loader_val, args.use_cuda,  n_last_blocks=args.n_last_blocks, avgpool_patchtokens=args.avgpool_patchtokens)

    #if utils.get_rank() == 0:
    #    train_features = nn.functional.normalize(train_features, dim=1, p=2)
    #    test_features = nn.functional.normalize(test_features, dim=1, p=2)

    # train_labels = torch.tensor([s[-1] for s in dataset_train]).long()
    # test_labels = torch.tensor([s[-1] for s in dataset_val.samples]).long()

    # save features and labels
    if args.dump_features and utils.get_rank() == 0:
        torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat.pth"))
        torch.save(test_features.cpu(), os.path.join(args.dump_features, "testfeat.pth"))
        torch.save(train_labels.cpu(), os.path.join(args.dump_features, "trainlabels.pth"))
        torch.save(test_labels.cpu(), os.path.join(args.dump_features, "testlabels.pth"))
    return train_features, test_features, train_labels, test_labels


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False, nb=None, n_last_blocks=4, avgpool_patchtokens=False):
    use_cuda = False
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = []
    labels_list = []
    # for samples in metric_logger.log_every(data_loader, 10):
    for samples, labels in data_loader:
        samples = samples.cuda(non_blocking=True)
        #if multiscale:
        #    feats = utils.multi_scale(samples, model)
        #else:
        #    feats = model(samples).clone()
        with torch.no_grad():
            intermediate_output = model.get_intermediate_layers(samples, n_last_blocks)
            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
            if avgpool_patchtokens:
                output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                output = output.reshape(output.shape[0], -1)
        feats = output
        feats = feats.data.cpu()
        features.append(feats.cpu())
        labels_list.append(labels)
        if nb and len(features) >= nb:
            break
    return torch.cat(features), torch.cat(labels_list)


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float,
        help='Temperature used in the voting coefficient')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_features', default=None,
        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--sklearn_model', default='logistic_regression', type=str)
    parser.add_argument('--nb_batches', default=None, type=int)
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--log_file', default="log.txt", type=str)
    parser.add_argument('--shots', default=0, type=int)
    parser.add_argument('--label_type', default="int", type=str)
    parser.add_argument('--dataset', default="image_folder", type=str)
    parser.add_argument('--nb_classes', default=10, type=int)
    parser.add_argument('--nb_repeats', default=1, type=int)
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
    for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    args = parser.parse_args()

    # utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    from glob import glob
    if os.path.isdir(args.pretrained_weights):
        paths = sorted(glob(os.path.join(args.pretrained_weights, "*.pth")))
    else:
        paths = [args.pretrained_weights]
    log_file = open(args.log_file, "w")
    for i, path in enumerate(sorted(paths)):
        print(path)
        accs = []
        for _ in range(args.nb_repeats):
            basename = os.path.basename(path)
            train_features, test_features, train_labels, test_labels = extract_feature_pipeline(args, path)
            train_features = train_features.numpy()
            train_labels = train_labels.numpy()
            
            mean = train_features.mean(axis=0, keepdims=True)
            std = train_features.std(axis=0, keepdims=True) + 1e-8

            test_features = test_features.numpy()
            test_labels = test_labels.numpy()
            
            train_features = (train_features - mean) / std
            test_features = (test_features - mean) / std
            
            if args.sklearn_model == "logistic_regression":
                clf = LogisticRegression(n_jobs=-1, verbose=1, C=args.alpha)
            elif args.sklearn_model == "linear_regression":
                clf =  Ridge(alpha=args.alpha)
            elif args.sklearn_model == "knn":
                clf = KNeighborsClassifier(n_neighbors=20)
            else:
                raise ValueError(args.sklearn_model)
            
            if args.sklearn_model == "linear_regression":
                nb_classes = train_labels.max() + 1
                train_labels_onehot = np.zeros((len(train_labels), nb_classes))
                train_labels_onehot[np.arange(len(train_labels)), train_labels] = 1.0
                clf.fit(train_features, train_labels_onehot)
                pred = clf.predict(test_features).argmax(axis=1)
                acc = (pred == test_labels).mean()
            else:
                clf.fit(train_features, train_labels)
                acc = (clf.predict(test_features) == test_labels).mean()
            print("ACCURACY", acc)
            accs.append(acc)
            if basename.startswith("checkpoint"):
                try:
                    epoch = int(basename[len("checkpoint"):].split(".")[0])
                except Exception:
                    epoch = None
            else:
                epoch = None
        acc = np.mean(accs)
        std = np.std(accs)
        dump = {
            "accuracy": accs,
            "mean_accuracy": np.mean(accs),
            "std_accuracy": np.std(accs),
            "weights": path,
            "epoch": epoch,
        }
        json.dump(dump, log_file)
        log_file.write("\n")
    # print(classification_report(test_labels, clf.predict(test_features), digits=4))
    log_file.close()
