from data import get_wds_dataset
from torchvision import transforms as pth_transforms
from glob import glob
import torchvision
from torchvision.utils import save_image
from torchvision.transforms import functional as TF

import utils

class args:
    dist_url = "env://" 

utils.init_distributed_mode(args)

print(utils.get_rank(), utils.get_world_size())
val_transform = pth_transforms.Compose([
    pth_transforms.Resize(256, interpolation=3),
    pth_transforms.CenterCrop(224),
    pth_transforms.ToTensor(),
#    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
train_transform = pth_transforms.Compose([
    pth_transforms.RandomResizedCrop(224),
    pth_transforms.RandomHorizontalFlip(),
    pth_transforms.ToTensor(),
#    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

class wds_args:
    distributed = True
    batch_size = 16
    train_data = glob("datasets/imagenet-1K/webdataset/train/**/*.tar")
    val_data = glob("datasets/imagenet-1K/webdataset/valid/**/*.tar")
    workers = 16
    num_batches = None
    train_num_samples = 1281167
    val_num_samples = 50000
    seed = utils.get_rank()
    input_col = "jpg"
    output_col = "cls"
    label_type = "int" 
    world_size = utils.get_world_size()
data_loader = get_wds_dataset(args=wds_args, preprocess_img=val_transform, preprocess_target=int, is_train=True).dataloader
print(data_loader)
nb = 0
for x, y in data_loader:
    nb += len(x)
    print(nb)
    grid = torchvision.utils.make_grid(x.cpu())
    TF.to_pil_image(grid).save(f"out{utils.get_rank()}.png")
    break
    
