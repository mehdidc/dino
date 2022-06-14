from glob import glob
import os
from clize import run
from subprocess import call

machine = open("/etc/FZJ/systemname").read().strip()
datasets = {
    "imagenet1k": '--data_path "/p/scratch/ccstdl/cherti1/imagenet-1K-lmdb/train" --label_type int --dataset caffe_lmdb',
    "imagenet21k": '--dataset caffe_lmdb_multiple --data_path datasets/imagenet-21K/lmdb',
    "laion400m": '--dataset wds --data_path "/p/scratch/ccstdl/katta1/LAION-400M/laion400m-dat-release/{00000..41455}.tar"',
    "cifar10": '--data_path datasets/cifar10 --dataset image_folder'
}

datasets_finetuning = {
    "imagenet1k": '--data_path "/p/scratch/ccstdl/cherti1/imagenet-1K-lmdb" --label_type int --dataset caffe_lmdb --nb_classes 1000',
    "cifar10": '--data_path datasets/cifar10 --dataset image_folder --nb_classes 10',
    "imagenet1k_wds": '--data_path /p/scratch/ccstdl/cherti1/imagenet-1K-webdataset --label_type int --dataset wds --nb_classes 1000 --train_num_samples 1281167 --val_num_samples 50000',
}

templates = {

    "vits16": {"arch": "vit_small", "patch_size": 16, "out_dim": 65536, "norm_last_layer": "false", "warmup_teacher_temp": 0.04, "teacher_temp": 0.07, "warmup_teacher_temp_epochs": 30, "use_fp16": "false", "weight_decay": 0.04, "weight_decay_end": 0.4, "clip_grad": 0, "batch_size_per_gpu": 64, "epochs": 800, "freeze_last_layer": 1, "lr": 0.0005, "warmup_epochs": 10, "min_lr": 1e-05, "global_crops_scale": [0.25, 1.0], "local_crops_scale": [0.05, 0.25], "local_crops_number": 10, "seed": 0, "num_workers": 10, "optimizer": "adamw", "momentum_teacher": 0.996, "use_bn_in_head": "false", "drop_path_rate": 0.1},
    
    "vits8": {"arch": "vit_small", "patch_size": 8, "out_dim": 65536, "norm_last_layer": "false", "warmup_teacher_temp": 0.04, "teacher_temp": 0.07, "warmup_teacher_temp_epochs": 30, "use_fp16": "false", "weight_decay": 0.04, "weight_decay_end": 0.4, "clip_grad": 3.0, "batch_size_per_gpu": 16, "epochs": 800, "freeze_last_layer": 1, "lr": 0.0005, "warmup_epochs": 10, "min_lr": 1e-06, "global_crops_scale": [0.4, 1.0], "local_crops_scale": [0.05, 0.4], "local_crops_number": 10, "seed": 0, "num_workers": 10, "optimizer": "adamw", "momentum_teacher": 0.996, "use_bn_in_head": "false", "drop_path_rate": 0.1},

    "vitb16": {"arch": "vit_base", "patch_size": 16, "out_dim": 65536, "norm_last_layer": "true", "warmup_teacher_temp": 0.04, "teacher_temp": 0.07, "warmup_teacher_temp_epochs": 50, "use_fp16": "false", "weight_decay": 0.04, "weight_decay_end": 0.4, "clip_grad": 0.3, "batch_size_per_gpu": 32, "epochs": 400, "freeze_last_layer": 3, "lr": 0.00075, "warmup_epochs": 10, "min_lr": 2e-06, "global_crops_scale": [0.25, 1.0], "local_crops_scale": [0.05, 0.25], "local_crops_number": 10, "seed": 0, "num_workers": 10, "optimizer": "adamw", "momentum_teacher": 0.996, "use_bn_in_head": "false", "drop_path_rate": 0.1},

    "vitb32": {"arch": "vit_base", "patch_size": 32, "out_dim": 65536, "norm_last_layer": "true", "warmup_teacher_temp": 0.04, "teacher_temp": 0.07, "warmup_teacher_temp_epochs": 50, "use_fp16": "false", "weight_decay": 0.04, "weight_decay_end": 0.4, "clip_grad": 0.3, "batch_size_per_gpu": 128, "epochs": 400, "freeze_last_layer": 3, "lr": 0.00075, "warmup_epochs": 10, "min_lr": 2e-06, "global_crops_scale": [0.25, 1.0], "local_crops_scale": [0.05, 0.25], "local_crops_number": 10, "seed": 0, "num_workers": 10, "optimizer": "adamw", "momentum_teacher": 0.996, "use_bn_in_head": "false", "drop_path_rate": 0.1},
    
    "vitb32_imagenet21k": {"arch": "vit_base", "patch_size": 32, "out_dim": 65536, "norm_last_layer": "true", "warmup_teacher_temp": 0.04, "teacher_temp": 0.07, "warmup_teacher_temp_epochs": 5, "use_fp16": "false", "weight_decay": 0.04, "weight_decay_end": 0.4, "clip_grad": 0.3, "batch_size_per_gpu": 128, "epochs": 40, "freeze_last_layer": 3, "lr": 0.00075, "warmup_epochs": 1, "min_lr": 2e-06, "global_crops_scale": [0.25, 1.0], "local_crops_scale": [0.05, 0.25], "local_crops_number": 10, "seed": 0, "num_workers": 20, "optimizer": "adamw", "momentum_teacher": 0.996, "use_bn_in_head": "false", "drop_path_rate": 0.1},

    "vitb8": {"arch": "vit_base", "patch_size": 8, "out_dim": 65536, "norm_last_layer": "true", "warmup_teacher_temp": 0.03, "teacher_temp": 0.07, "warmup_teacher_temp_epochs": 50, "use_fp16": "false", "weight_decay": 0.04, "weight_decay_end": 0.4, "clip_grad": 3.0, "batch_size_per_gpu": 6, "epochs": 300, "freeze_last_layer": 3, "lr": 0.0005, "warmup_epochs": 10, "min_lr": 2e-06, "global_crops_scale": [0.25, 1.0], "local_crops_scale": [0.05, 0.25], "local_crops_number": 10, "seed": 0, "num_workers": 10, "optimizer": "adamw", "momentum_teacher": 0.996, "use_bn_in_head": "false", "drop_path_rate": 0.1},


    "laion400m_vitb32": {"arch": "vit_base", "patch_size": 32, "out_dim": 65536, "norm_last_layer": "false", "warmup_teacher_temp": 0.04, "teacher_temp": 0.07, "warmup_teacher_temp_epochs": 1, "use_fp16": "false", "weight_decay": 0.04, "weight_decay_end": 0.4, "clip_grad": 0.3, "batch_size_per_gpu": 256, "epochs": 32, "freeze_last_layer": 3, "lr": 0.0003, "warmup_epochs": 1, "min_lr": 0.000002, "global_crops_scale": [0.25, 1.0], "local_crops_scale": [0.05, 0.25], "local_crops_number": 1, "seed": 0, "num_workers": 20, "optimizer": "lamb", "momentum_teacher": 0.996, "use_bn_in_head": "false", "drop_path_rate": 0.1},
    
    "vitl16": {"arch": "vit_large", "patch_size": 16, "out_dim": 65536, "norm_last_layer": "true", "warmup_teacher_temp": 0.04, "teacher_temp": 0.07, "warmup_teacher_temp_epochs": 50, "use_fp16": "false", "weight_decay": 0.04, "weight_decay_end": 0.4, "clip_grad": 0.3, "batch_size_per_gpu": 32, "epochs": 400, "freeze_last_layer": 3, "lr": 0.00075, "warmup_epochs": 10, "min_lr": 2e-06, "global_crops_scale": [0.25, 1.0], "local_crops_scale": [0.05, 0.25], "local_crops_number": 10, "seed": 0, "num_workers": 10, "optimizer": "adamw", "momentum_teacher": 0.996, "use_bn_in_head": "false", "drop_path_rate": 0.1},
}
def train(
    *,
    dataset="imagenet1k",
    template="vits16",
    folder="out",
    nodes=16,
    gpus_per_node=4,
    t=360,
    batch_size_per_gpu:int=None,
    optimizer:str=None,
    local_crops_number:int=None,
    lr:float=None,
    min_lr:float=None,
    epochs:int=None,
    num_workers:int=None,
    saveckp_freq:int=None,
    num_samples_per_epoch:int=None,
):
    os.makedirs(folder, exist_ok=True)
    #nb_runs = len(glob(os.path.join(folder, "out_*")))
    #run_id = nb_runs + 1
    output = f"{folder}/slurm-%j.out"
    error = f"{folder}/slurm-%j.err"
    script = f"scripts/run_{machine}_ddp.sh main_dino.py"
    data = datasets[dataset]
    hypers = templates[template]
    
    if batch_size_per_gpu is not None:
        hypers["batch_size_per_gpu"] = batch_size_per_gpu
    if lr is not None:
        hypers["lr"] = lr
    if min_lr is not None:
        hypers["min_lr"] = min_lr
    if epochs is not None:
        hypers["epochs"] = epochs
    if optimizer is not None:
        hypers["optimizer"] = optimizer
    if local_crops_number is not None:
        hypers["local_crops_number"] = local_crops_number
    if num_workers is not None:
        hypers["num_workers"] = num_workers
    if saveckp_freq is not None:
        hypers["saveckp_freq"] = saveckp_freq
    if num_samples_per_epoch is not None:
        hypers["num_samples_per_epoch"] = num_samples_per_epoch
    def to_str(v):
        if type(v) == list:
            return " ".join(map(str, v))
        else:
            return v
    hypers_str = " ".join(f"--{k} {to_str(v)}" for k, v in hypers.items())
    cmd = f"sbatch --gres=gpu:{gpus_per_node} -t {t} --output {output} --error {error} -N {nodes} -n {nodes*gpus_per_node} {script} {data} {hypers_str} --output_dir {folder}"
    print(cmd)
    call(cmd,shell=True)

def linear_probe(
    *,
    dataset="cifar10",
    template="vits16",
    folder="out",
    batch_size_per_gpu=128,
    nodes=2,
    gpus_per_node=4,
    shots=0,
    weights:str=None,
    t=360,
    epochs=100,
    n_last_blocks=4,
    avgpool_patchtokens=False,
    batch_norm=False,
    target:str=None,
    num_workers=10,
    lr:float=0.001,
):
    hypers = templates[template]
    arch = hypers["arch"]
    data = datasets_finetuning[dataset]
    patch_size = hypers["patch_size"]
    fs = f"_fs{shots}" if shots else ""
    script = f"scripts/run_{machine}_ddp.sh eval_linear.py"
    if weights is None:
        weights = os.path.join(folder, "checkpoint.pth")
    
    if target is not None:
        folder = target
    else:
        folder = os.path.join(folder, f"eval_linear_{dataset}")
    output = f"{folder}/slurm-%j.out"
    error = f"{folder}/slurm-%j.err"
    bn = "--batch_norm" if batch_norm else ""
    os.makedirs(folder, exist_ok=True)
    cmd = f"sbatch --gres=gpu:{gpus_per_node} -t {t} --output {output} --error {error} -N {nodes} -n {nodes*gpus_per_node} {script} {data}  --arch {arch} --patch_size {patch_size} --batch_size_per_gpu {batch_size_per_gpu} --pretrained_weights {weights} --output_dir {folder} --n_last_blocks {n_last_blocks} --avgpool_patchtokens {avgpool_patchtokens} --epochs {epochs} {bn} --num_workers {num_workers} --lr {lr}"
    print(cmd)
    call(cmd, shell=True)

def fast_linear_probe(
    *,
    dataset="cifar10",
    template="vits16",
    folder="out",
    batch_size_per_gpu=256,
    shots=0,
    sklearn_model="logistic_regression",
    alpha=1.0,
    nb_repeats=1,
    n_last_blocks=4,
    avgpool_patchtokens=False,
):
    hypers = templates[template]
    arch = hypers["arch"]
    data = datasets_finetuning[dataset]
    patch_size = hypers["patch_size"]
    fs = f"_fs{shots}" if shots else ""
    target = os.path.join(folder, f"eval_fast_linear_{dataset}{fs}")
    os.makedirs(target, exist_ok=True)

    script = f"scripts/run_{machine}_ddp.sh eval_sklearn.py"
    cmd = f'sbatch --output {target}/slurm-%j.out --error {target}/slurm-%j.err  -N 1 -n 1 {script}  --arch {arch} --patch_size {patch_size} --pretrained_weights {folder} {data} --batch_size_per_gpu {batch_size_per_gpu} --log {target}/log.txt --shots {shots} --sklearn_model {sklearn_model} --alpha {alpha} --nb_repeats {nb_repeats} --n_last_blocks {n_last_blocks} --avgpool_patchtokens {avgpool_patchtokens}'
    print(cmd)
    call(cmd, shell=True)

if __name__ == "__main__":
    run([train, fast_linear_probe, linear_probe])

