import os
import sys
import torch
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.utils.data as data
import torch.nn.functional as F
from Dataset.init_dataset import genimage, celebdf2
from torch.autograd import Variable
from Network.Models_Def.resnet import ResNet10,ResNet12
from Network.Models_Def.shufflenet import ShuffleNetG2
from Network.Models_Def.mobilnet_v2 import MobileNetV2
import torchvision.transforms as transforms
import random
import torch.nn as nn
from torchvision.models import efficientnet_b0, efficientnet_b4

Seed = 0
seed = Seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Project root (parent of Dataset/): stable regardless of cwd when running scripts.
_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_UTIL_DIR, '..'))
LOG_DIR = os.path.join(PROJECT_ROOT, 'Logs')
Project_Path = os.path.join(PROJECT_ROOT, '')


def init_logs(log_level=logging.INFO, sub_name=None):
    """Attach file + console logging. Log files are always written under ``Logs`` at project root."""
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    mkdirs(LOG_DIR)
    filename = os.path.basename(sys.argv[0][0:-3])
    if sub_name is None:
        logfile = os.path.join(LOG_DIR, filename + '.log')
    else:
        logfile = os.path.join(LOG_DIR, filename + '_' + sub_name + '.log')
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    console  = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(console)
    return logger

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


def generate_public_data_indexs(dataset,datadir,size, noise_type=None, noise_rate=0):
    if dataset =='cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir, noise_type=noise_type, noise_rate=noise_rate)
    n_train = y_train.shape[0]
    idxs = np.random.permutation(n_train)
    idxs = idxs[0:size]
    return idxs

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0, noise_type=None, noise_rate=0, train_flag=False):
    if dataset in ('genimage', 'celebdf2'):
        if dataset == 'genimage':
            dl_obj = genimage
            transform_train = transforms.Compose([
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.Resize((224, 224)),
                # transforms.Resize((64, 64)),
                transforms.ToTensor(),
                # transforms.ToPILImage(),
                # transforms.ColorJitter(brightness=noise_level),
                # transforms.RandomCrop(32),
                # transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.Resize((224, 224)),
                # transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
                ])
        elif dataset == 'celebdf2':
            dl_obj = celebdf2
            transform_train = transforms.Compose([
                # transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                # transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                ])
        if train_flag == True:
            train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, noise_type=noise_type, noise_rate=noise_rate)
            train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True)
            return train_dl, train_ds
        else:
            test_ds = dl_obj(datadir, train=False, transform=transform_test)
            test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)
            return test_dl, test_ds
        
def init_nets(n_parties,nets_name_list, num_classes=2):
    nets_list = {net_i: None for net_i in range(n_parties)}
    for net_i in range(n_parties):
        net_name = nets_name_list[net_i]
        if net_name=='ResNet10':
            net = ResNet10(num_classes=num_classes)
        elif net_name =='ResNet12':
            net = ResNet12(num_classes=num_classes)
        elif net_name =='ShuffleNet':
            net = ShuffleNetG2(num_classes=num_classes)
        elif net_name =='Mobilenetv2':
            net = MobileNetV2(num_classes=num_classes)
        elif net_name =='EfficientNetB0':
            net = efficientnet_b0(pretrained=True)
            net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)
        elif net_name =='EfficientNetB4':
            net = efficientnet_b4(pretrained=True)
            net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)
        nets_list[net_i] = net
    return nets_list

if __name__ =='__main__':
    logger = init_logs()
    public_data_indexs = generate_public_data_indexs(dataset='genimage',datadir='./genimage',size=5000)
    train_dl, test_dl, train_ds, test_ds = get_dataloader(dataset='genimage',datadir='./genimage',train_bs=256,test_bs=512,dataidxs=public_data_indexs)
    print(len(train_ds))
