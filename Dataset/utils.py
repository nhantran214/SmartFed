import os
import sys
import csv
import json
import torch
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.utils.data as data
import torch.nn.functional as F
from datetime import datetime
from Dataset.init_dataset import genimage, celebdf2
from torch.autograd import Variable
from Network.Models_Def.resnet import ResNet10,ResNet12
from Network.Models_Def.shufflenet import ShuffleNetG2
from Network.Models_Def.mobilnet_v2 import MobileNetV2
import torchvision.transforms as transforms
import random
import torch.nn as nn
from torchvision.models import efficientnet_b0, efficientnet_b4
from sklearn.metrics import roc_auc_score

Seed = 0
seed = Seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Project root (parent of Dataset/): stable regardless of cwd when running scripts.
_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_UTILS_DIR, '..'))
LOG_DIR = os.path.join(PROJECT_ROOT, 'Logs')
Project_Path = os.path.join(PROJECT_ROOT, '')


class FlushFileHandler(logging.FileHandler):
    """FileHandler that flushes after every emit so logs survive crashes."""

    def emit(self, record):
        super().emit(record)
        self.flush()


def init_logs(log_level=logging.INFO, sub_name=None, run_id=None):
    """Attach timestamped file + console logging under ``Logs`` at project root.

    Creates:
      - ``Logs/<script>[_sub]_<timestamp>.log`` (unique per run)
      - ``Logs/<script>[_sub].log`` (latest run, overwritten)
    """
    logger = logging.getLogger('smartfed')
    logger.setLevel(log_level)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()

    mkdirs(LOG_DIR)
    script = os.path.basename(sys.argv[0][0:-3]) if sys.argv and sys.argv[0] else 'train'
    base = f'{script}_{sub_name}' if sub_name else script
    if run_id is None:
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    logfile_run = os.path.join(LOG_DIR, f'{base}_{run_id}.log')
    logfile_latest = os.path.join(LOG_DIR, f'{base}.log')

    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    )

    fh_run = FlushFileHandler(logfile_run, mode='w', encoding='utf-8')
    fh_run.setLevel(log_level)
    fh_run.setFormatter(formatter)

    fh_latest = FlushFileHandler(logfile_latest, mode='w', encoding='utf-8')
    fh_latest.setLevel(log_level)
    fh_latest.setFormatter(formatter)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(formatter)

    logger.addHandler(fh_run)
    logger.addHandler(fh_latest)
    logger.addHandler(console)

    logger.info(f'Logging to file: {logfile_run}')
    logger.info(f'Also writing latest log: {logfile_latest}')
    logger.run_id = run_id
    logger.log_dir = LOG_DIR
    logger.logfile = logfile_run
    return logger


class MetricsLogger:
    """Append training/eval metrics to a CSV under ``Logs``."""

    FIELDNAMES = [
        'timestamp', 'run_id', 'phase', 'client_id', 'epoch', 'round',
        'loss', 'accuracy', 'auc', 'extra',
    ]

    def __init__(self, run_id=None, filename=None):
        mkdirs(LOG_DIR)
        self.run_id = run_id or datetime.now().strftime('%Y%m%d_%H%M%S')
        script = os.path.basename(sys.argv[0][0:-3]) if sys.argv and sys.argv[0] else 'train'
        self.path = filename or os.path.join(LOG_DIR, f'{script}_metrics_{self.run_id}.csv')
        self._fh = open(self.path, 'w', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self._fh.flush()

    def log(self, phase, client_id=None, epoch=None, round=None,
            loss=None, accuracy=None, auc=None, extra=None):
        row = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'run_id': self.run_id,
            'phase': phase,
            'client_id': '' if client_id is None else client_id,
            'epoch': '' if epoch is None else epoch,
            'round': '' if round is None else round,
            'loss': '' if loss is None else f'{float(loss):.6f}',
            'accuracy': '' if accuracy is None else f'{float(accuracy):.4f}',
            'auc': '' if auc is None else f'{float(auc):.6f}',
            'extra': '' if extra is None else str(extra),
        }
        self._writer.writerow(row)
        self._fh.flush()

    def close(self):
        if getattr(self, '_fh', None):
            self._fh.close()
            self._fh = None

    def __del__(self):
        self.close()


def log_run_config(logger, config_cls):
    """Write key Config fields into the text log at run start."""
    logger.info('===== Run configuration =====')
    keys = [
        'SEED', 'PROTOCOL', 'DATASET_TYPE', 'N_CLIENTS', 'N_PARTIES', 'HAS_SERVER_TEACHER',
        'SERVER_TEACHER_ID', 'CLIENT_IDS', 'PRETRAIN_PARTY_IDS', 'GSAM_UPDATE_SERVER',
        'COMMUNICATION_EPOCH', 'PRETRAIN_EPOCH',
        'AGENT_MODE', 'DCSA_MODE', 'AFWA_MODE', 'DCSA_RULE_STRATEGY',
        'SELECTION_BUDGET', 'AFWA_FIXED_COEFFICIENT', 'AFWA_COEFF_MIN', 'AFWA_COEFF_MAX',
        'WEIGHT_RULE',
        'LOG_LLM_COST', 'OLLAMA_MODEL_NAME',
        'TRAIN_BATCH_SIZE', 'TEST_BATCH_SIZE',
        'TRAIN_BATCH_SIZE_PRIVATE', 'TEST_BATCH_SIZE_PRIVATE',
        'PARALLEL_CLIENTS', 'PARALLEL_EVAL',
        'REQUIRE_CUDA', 'KEEP_MODELS_ON_GPU', 'CUDNN_BENCHMARK',
        'OLLAMA_ON_GPU', 'OLLAMA_UNLOAD_AFTER_QUERY', 'OLLAMA_TIMEOUT',
        'OLLAMA_MAX_RETRIES', 'OLLAMA_MAX_TOKENS',
        'LOCAL_WEIGHT_MAX_BATCHES',
        'EVAL_BATCH_SIZE', 'EVAL_USE_AMP',
        'EVAL_MAX_BATCHES', 'EVAL_EVERY_N_ROUNDS', 'EVAL_OWN_DATASET_ONLY', 'EVAL_FINAL_FULL',
        'DATA_CACHE_ENABLED', 'DATA_CACHE_DIR', 'DATA_CACHE_IMAGE_SIZE', 'DATA_CACHE_PREWARM',
        'PUBLIC_KL_MAX_BATCHES', 'PUBLIC_KL_LR', 'PUBLIC_KL_TEMPERATURE',
        'PUBLIC_KL_CONSENSUS', 'PUBLIC_KL_HEAD_ONLY', 'PUBLIC_KL_SKIP_TEACHER0',
        'PUBLIC_KL_USE_PUBLIC_MODEL',
        'GSAM_TOTAL_EPOCHS', 'GSAM_MAX_BATCHES', 'GSAM_RHO_MAX', 'GSAM_RHO_MIN', 'GSAM_ALPHA', 'GSAM_LR',
        'PRIVATE_DATASET_DIR', 'PUBLIC_DATASET_DIR', 'MODEL_STORAGE_ROOT', 'MODEL_STORAGE_MAIN_ROOT',
        'HETE_BACKBONE_POOL', 'PRIVATE_NETS_NAME_LIST', 'NETS_NAME_LIST', 'PARTICIPANT_PARAMS',
        'NOISE_TYPE', 'NOISE_RATE', 'INTRA_DATASET',
    ]
    for key in keys:
        if hasattr(config_cls, key):
            logger.info(f'  {key} = {getattr(config_cls, key)}')
    logger.info('===== End configuration =====')


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


def across_is_better(accuracy, auc, full_eval, best):
    """Prefer a full validation over a subsampled one. Then higher accuracy, then AUC.

    ``best`` is a dict with keys acc, auc, round, full. ``round is None`` means unset.
    """
    if best.get('round') is None:
        return True
    if full_eval and not best.get('full'):
        return True
    if best.get('full') and not full_eval:
        return False
    auc_cmp = auc if auc == auc else -1.0
    best_auc = best.get('auc', -1.0)
    best_auc = best_auc if best_auc == best_auc else -1.0
    if accuracy > best.get('acc', -1.0):
        return True
    if accuracy == best.get('acc', -1.0) and auc_cmp > best_auc:
        return True
    return False


def save_best_checkpoints(models, nets_name_list, base_dir, epoch_index, accuracy, auc, full_eval, logger):
    """Overwrite ``*_best.ckpt`` and ``best_metrics.json`` for the current best round."""
    mkdirs(base_dir)
    for i, model in enumerate(models):
        netname = nets_name_list[i]
        state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(state, os.path.join(base_dir, f'{netname}_{i}_best.ckpt'))
    auc_out = None if auc != auc else float(auc)
    meta = {
        'round': int(epoch_index),
        'accuracy': float(accuracy),
        'auc': auc_out,
        'full_eval': bool(full_eval),
    }
    with open(os.path.join(base_dir, 'best_metrics.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh)
    logger.info(
        f'New best across-eval at round {epoch_index}: '
        f'Acc={accuracy:.2f}%, AUC={auc:.4f}, full_eval={full_eval}. '
        f'Saved *_best.ckpt under {base_dir}'
    )


def load_checkpoint(model, path, map_location='cpu'):
    """Load a checkpoint into ``model``, tolerating DataParallel key prefixes.

    Supports:
      - plain keys: ``features...`` (current pretrain saves)
      - DataParallel keys: ``module.features...`` (legacy saves)
    """
    state = torch.load(path, map_location=map_location)
    if not isinstance(state, dict):
        raise TypeError(f'Unexpected checkpoint type at {path}: {type(state)}')

    # Some checkpoints wrap weights under a nested dict.
    if 'state_dict' in state and isinstance(state['state_dict'], dict):
        state = state['state_dict']

    model_is_dp = isinstance(model, nn.DataParallel)
    keys = list(state.keys())
    ckpt_is_dp = len(keys) > 0 and keys[0].startswith('module.')

    if model_is_dp and not ckpt_is_dp:
        state = {f'module.{k}': v for k, v in state.items()}
    elif (not model_is_dp) and ckpt_is_dp:
        state = {k[len('module.'):]: v for k, v in state.items()}

    model.load_state_dict(state)
    return model


def compute_accuracy_and_auc(model, dataloader, device, max_batches=None, use_amp=None):
    """Compute accuracy (%) and AUC on a dataloader.

    Binary tasks use the positive-class probability; multiclass uses
    one-vs-rest macro AUC.

    Args:
        max_batches: If set, only evaluate this many batches (quick probe).
        use_amp: FP16 autocast on CUDA. Default from Config.EVAL_USE_AMP.
    """
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_probs = []

    if use_amp is None:
        try:
            from config import Config as _Cfg
            use_amp = bool(getattr(_Cfg, 'EVAL_USE_AMP', False))
        except Exception:
            use_amp = False
    device_obj = torch.device(device) if not isinstance(device, torch.device) else device
    use_amp = bool(use_amp) and device_obj.type == 'cuda'

    try:
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                images = images.to(device_obj, non_blocking=True)
                labels = labels.to(device_obj, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.append(labels.detach().cpu().numpy())
                all_probs.append(probs.detach().float().cpu().numpy())

        accuracy = 100.0 * correct / total if total > 0 else 0.0
        if total == 0:
            return accuracy, float('nan')

        y_true = np.concatenate(all_labels)
        y_prob = np.concatenate(all_probs)
        n_classes = y_prob.shape[1]

        try:
            if n_classes == 2:
                auc = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                auc = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))
        except ValueError:
            # e.g. only one class present in the batch/labels
            auc = float('nan')

        return accuracy, auc
    finally:
        # Critical: restore train/eval mode so later epochs do not train under BN eval().
        model.train(was_training)


def split_cifar10_indices(
    n_clients,
    seed,
    public_train=5000,
    public_test=1000,
    total_train=50000,
    total_test=10000,
    client_ids=None,
):
    """IID split: public hold-out first, remainder equally among clients.

    Last client receives leftover samples so counts always sum to the pool.
    """
    n_clients = int(n_clients)
    public_train = int(public_train)
    public_test = int(public_test)
    if n_clients < 1:
        raise ValueError(f'n_clients must be >= 1, got {n_clients}')
    if not (0 <= public_train < total_train):
        raise ValueError(
            f'public_train={public_train} out of range for total_train={total_train}'
        )
    if not (0 <= public_test < total_test):
        raise ValueError(
            f'public_test={public_test} out of range for total_test={total_test}'
        )
    if client_ids is None:
        client_ids = list(range(n_clients))
    if len(client_ids) != n_clients:
        raise ValueError(
            f'client_ids length {len(client_ids)} != n_clients={n_clients}'
        )

    rng = np.random.RandomState(int(seed))
    train_indices = rng.permutation(total_train)
    test_indices = rng.permutation(total_test)

    public_train_dataidx = train_indices[:public_train]
    local_train_indices = train_indices[public_train:]
    public_test_dataidx = test_indices[:public_test]
    local_test_indices = test_indices[public_test:]

    local_train_n = len(local_train_indices)
    local_test_n = len(local_test_indices)
    train_per = local_train_n // n_clients
    test_per = local_test_n // n_clients

    net_dataidx_map = {}
    test_dataidx_map = {}
    for li, index in enumerate(client_ids):
        t0 = li * train_per
        t1 = local_train_n if li == n_clients - 1 else t0 + train_per
        net_dataidx_map[index] = local_train_indices[t0:t1]
        v0 = li * test_per
        v1 = local_test_n if li == n_clients - 1 else v0 + test_per
        test_dataidx_map[index] = local_test_indices[v0:v1]

    return public_train_dataidx, public_test_dataidx, net_dataidx_map, test_dataidx_map


def generate_public_data_indexs(dataset,datadir,size, noise_type=None, noise_rate=0):
    if dataset =='cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir, noise_type=noise_type, noise_rate=noise_rate)
    n_train = y_train.shape[0]
    idxs = np.random.permutation(n_train)
    idxs = idxs[0:size]
    return idxs

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0, noise_type=None, noise_rate=0, train_flag=False, num_workers=None):
    if dataset in ('genimage', 'diffusionface', 'celebdf2'):
        # diffusionface reuses the GenImage folder layout: train|val / nature|ai
        if dataset in ('genimage', 'diffusionface'):
            dl_obj = genimage
            transform_train = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
                ])
        elif dataset == 'celebdf2':
            dl_obj = celebdf2
            transform_train = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                ])

        if num_workers is None:
            try:
                from config import Config as _Cfg
                num_workers = int(getattr(_Cfg, 'DATALOADER_NUM_WORKERS', 4))
            except Exception:
                num_workers = 4
        # pin_memory helps host→GPU copy; persistent_workers avoids respawning
        # workers every epoch when the same DataLoader is reused.
        loader_kwargs = dict(
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0),
        )
        if num_workers > 0:
            loader_kwargs['prefetch_factor'] = 2

        if train_flag == True:
            train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, noise_type=noise_type, noise_rate=noise_rate)
            train_dl = data.DataLoader(
                dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True, **loader_kwargs
            )
            return train_dl, train_ds
        else:
            test_ds = dl_obj(datadir, train=False, transform=transform_test)
            # Shuffle val when subsampling (EVAL_MAX_BATCHES): dataset is ordered
            # nature→ai, so the first N batches would otherwise be a single class.
            try:
                from config import Config as _Cfg
                shuffle_val = getattr(_Cfg, 'EVAL_MAX_BATCHES', None) is not None
            except Exception:
                shuffle_val = False
            test_dl = data.DataLoader(
                dataset=test_ds, batch_size=test_bs, shuffle=shuffle_val, **loader_kwargs
            )
            return test_dl, test_ds
        
class _LogitsOnlyWrapper(nn.Module):
    """ResNet/MobileNet/ShuffleNet return (logits, embedding); FL code expects logits only."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def init_nets(n_parties, nets_name_list, num_classes=2):
    if len(nets_name_list) < n_parties:
        raise ValueError(
            f'nets_name_list length {len(nets_name_list)} < n_parties={n_parties}'
        )
    nets_list = {net_i: None for net_i in range(n_parties)}
    for net_i in range(n_parties):
        net_name = nets_name_list[net_i]
        if net_name == 'ResNet10':
            net = _LogitsOnlyWrapper(ResNet10(num_classes=num_classes))
        elif net_name == 'ResNet12':
            net = _LogitsOnlyWrapper(ResNet12(num_classes=num_classes))
        elif net_name == 'ShuffleNet':
            net = _LogitsOnlyWrapper(ShuffleNetG2(num_classes=num_classes))
        elif net_name in ('Mobilenetv2', 'MobileNetV2', 'MobileNetv2'):
            net = _LogitsOnlyWrapper(MobileNetV2(num_classes=num_classes))
        elif net_name == 'EfficientNetB0':
            net = efficientnet_b0(pretrained=True)
            net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)
        elif net_name == 'EfficientNetB4':
            net = efficientnet_b4(pretrained=True)
            net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)
        else:
            raise ValueError(
                f'Unknown backbone {net_name!r}. '
                f'Supported: EfficientNetB0, EfficientNetB4, ResNet10, ResNet12, '
                f'Mobilenetv2, ShuffleNet'
            )
        nets_list[net_i] = net
    return nets_list

if __name__ =='__main__':
    logger = init_logs()
    public_data_indexs = generate_public_data_indexs(dataset='genimage',datadir='./genimage',size=5000)
    train_dl, test_dl, train_ds, test_ds = get_dataloader(dataset='genimage',datadir='./genimage',train_bs=256,test_bs=512,dataidxs=public_data_indexs)
    print(len(train_ds))
