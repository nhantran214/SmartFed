import sys
sys.path.append("..")
sys.path.append('./')
# Increasing recursion limit
sys.setrecursionlimit(10000)

# Standard library imports
import os
import random
import numpy as np
from numpy import mean, exp

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn

# Local imports
from Dataset.utils import (
    init_logs, get_dataloader, init_nets, generate_public_data_indexs, mkdirs,
    compute_accuracy_and_auc, MetricsLogger, log_run_config, load_checkpoint,
    split_cifar10_indices, across_is_better, save_best_checkpoints,
)
from Dataset.integrity import check_genimage_client_val
from Dataset.cifar10_dataset import CIFAR10Dataset, get_cifar10_transforms
from loss import SCELoss
from optimize import GSAMOptimizer
from config import Config
from llm_client_selection import (
    LLMClientSelectionSystem, create_llm_client_selector,
    OllamaLLMClient, ollama_inference_session, GLOBAL_LLM_COST,
)
from loss_weighting_agent import LossWeightingAgent, create_loss_weighting_agent, LossWeightingContext
from parallel_utils import iter_chunks, map_parallel
import json

def evaluate_model(model, dataloader, logger, device, max_batches=None):
    """Evaluate model performance on a dataloader (Accuracy + AUC)."""
    acc, auc = compute_accuracy_and_auc(
        model, dataloader, device, max_batches=max_batches
    )
    logger.info(f'Test Accuracy: {acc:.2f}%, AUC: {auc:.4f}')
    return acc, auc

def update_model_via_private_data(model, private_epoch, private_dataloader, loss_function, 
                                optimizer_method, learning_rate, logger, device):
    """Update model using private data"""
    # Initialize loss function
    if loss_function == 'CE':
        criterion = nn.CrossEntropyLoss()
    elif loss_function == 'SCE':
        criterion = SCELoss(alpha=0.1, beta=1.0, num_classes=10)
    
    # Initialize optimizer
    if optimizer_method == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_method == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    
    participant_local_loss_batch_list = []
    
    for epoch_index in range(private_epoch):
        for batch_idx, (images, labels) in enumerate(private_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            participant_local_loss_batch_list.append(loss.item())
            loss.backward()
            optimizer.step()
            
            if epoch_index % 5 == 0:
                logger.info(f'Private Train: [{batch_idx * len(images)}/{len(private_dataloader.dataset)} '
                          f'({100. * batch_idx / len(private_dataloader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    return model, participant_local_loss_batch_list

def train_gsam(model, train_dataloader, eval_dataloader, criterion, optimizer, 
              total_epochs, logger, device, start_epoch=0, on_epoch_end=None):
    """Train model using GSAM optimizer (two forwards per batch; no anomaly mode)."""
    local_loss_list = []
    
    for epoch in range(int(start_epoch), total_epochs):
        model.train()
        running_loss = 0.0
        n_seen = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            max_batches = getattr(Config, 'GSAM_MAX_BATCHES', None)
            if max_batches is not None and batch_idx >= int(max_batches):
                logger.info(
                    f'GSAM epoch {epoch + 1}: stop after {int(max_batches)} batches'
                )
                break
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            # GSAMOptimizer.step runs ascent + second forward/backward internally.
            loss_value = optimizer.step(inputs, labels, criterion, current_epoch=epoch)
            local_loss_list.append(loss_value)
            running_loss += loss_value * inputs.size(0)
            n_seen += inputs.size(0)
        
        epoch_loss = running_loss / max(n_seen, 1)
        logger.info(f"Epoch [{epoch+1}/{total_epochs}], Training Loss: {epoch_loss:.4f}")
        
        # Evaluation (subsample during GSAM; full eval is done in across_evaluation)
        gsam_eval_batches = getattr(Config, 'EVAL_MAX_BATCHES', None)
        accuracy, auc = compute_accuracy_and_auc(
            model, eval_dataloader, device, max_batches=gsam_eval_batches
        )
        logger.info(f'Evaluate Accuracy: {accuracy:.2f}%, AUC: {auc:.4f}')
        if on_epoch_end is not None:
            on_epoch_end(model, epoch)
    
    return model, local_loss_list

def across_evaluation(pretrained_model, N_CLIENTS, Private_Dataset_Name, Private_Dataset_Dir, 
                     TrainBatchSize, TestBatchSize, Noise_type, Noise_rate, logger, device,
                     test_dataidx_map=None, metrics_logger=None, round_idx=None,
                     cached_test_dls=None, full_eval=False):
    """Evaluate client models across client datasets (not the server teacher).

    Speed knobs (Config):
      - EVAL_MAX_BATCHES: subsample each val loader (ignored if full_eval / EVAL_FINAL_FULL)
      - EVAL_OWN_DATASET_ONLY: client party p → its own dataset only
    """
    max_batches = None
    if full_eval and getattr(Config, 'EVAL_FINAL_FULL', True):
        max_batches = None
    else:
        max_batches = getattr(Config, 'EVAL_MAX_BATCHES', None)
    own_only = bool(getattr(Config, 'EVAL_OWN_DATASET_ONLY', False))
    client_ids = list(getattr(Config, 'CLIENT_IDS', list(range(N_CLIENTS))))
    logger.info(
        f'Across Evaluate Models (clients={client_ids}, '
        f'max_batches={max_batches}, own_only={own_only}, full_eval={full_eval})'
    )

    eval_bs = int(getattr(Config, 'EVAL_BATCH_SIZE', Config.TEST_BATCH_SIZE_PRIVATE))
    # test_list[i] aligns with client_ids[i]
    if cached_test_dls is not None:
        test_list = [cached_test_dls[pid] for pid in client_ids]
    else:
        test_list = []
        for pid in client_ids:
            if Private_Dataset_Name in ('genimage', 'diffusionface'):
                root_genim_full = Config.party_datadir(pid)
                test_dl, _ = get_dataloader(dataset=Private_Dataset_Name, 
                                           datadir=root_genim_full,
                                           train_bs=eval_bs,
                                           test_bs=eval_bs,
                                           dataidxs=None,
                                           noise_type=Noise_type,
                                           noise_rate=Noise_rate)
            elif Private_Dataset_Name == 'cifar10':
                if test_dataidx_map is not None:
                    test_dataidx = test_dataidx_map[pid]
                else:
                    test_dataidx = None
                test_dl, _ = get_cifar10_dataloader(datadir=Private_Dataset_Dir,
                                                  train_bs=eval_bs,
                                                  test_bs=eval_bs,
                                                  dataidxs=test_dataidx,
                                                  train_flag=False)
            else:
                test_dl, _ = get_dataloader(dataset=Private_Dataset_Name, 
                                           datadir=Private_Dataset_Dir,
                                           train_bs=eval_bs,
                                           test_bs=eval_bs,
                                           dataidxs=None,
                                           noise_type=Noise_type,
                                           noise_rate=Noise_rate)
            test_list.append(test_dl)

    def evaluate_one_model(m):
        """m is a client party id (e.g. 1..8)."""
        model = pretrained_model[m].to(device)
        acc_model_list = []
        auc_model_list = []
        if own_only:
            dataset_positions = [client_ids.index(m)]
        else:
            dataset_positions = list(range(len(client_ids)))
        for i in dataset_positions:
            ds_party = client_ids[i]
            accuracy, auc = evaluate_model(
                model=model, dataloader=test_list[i], logger=logger, device=device,
                max_batches=max_batches,
            )
            acc_model_list.append(accuracy)
            auc_model_list.append(auc)
            logger.info(
                f'Test Accuracy of model {m} ({Config.party_dataset_key(m)}) '
                f'on dataset {ds_party} ({Config.party_dataset_key(ds_party)}): '
                f'{accuracy:.2f}%, AUC: {auc:.4f}'
            )
        accuracy_avg = sum(acc_model_list) / len(acc_model_list)
        auc_avg = float(np.nanmean(auc_model_list))
        logger.info(
            f'Average Test Accuracy of model {m}: {accuracy_avg:.2f}%, '
            f'Average AUC: {auc_avg:.4f}'
        )
        if metrics_logger is not None:
            metrics_logger.log(
                phase='across_eval',
                client_id=m,
                round=round_idx,
                accuracy=accuracy_avg,
                auc=auc_avg,
            )
        pretrained_model[m] = stash_model(model, device)
        return m, accuracy_avg, auc_avg

    scores = []
    for chunk in iter_chunks(client_ids, getattr(Config, 'PARALLEL_EVAL', 1)):
        logger.info(f'Across-eval model chunk: {list(chunk)}')
        scores.extend(map_parallel(evaluate_one_model, chunk, max_workers=len(chunk)))
        if not _keep_models_on_gpu():
            torch.cuda.empty_cache()
    if not scores:
        return None, None, max_batches is None
    mean_acc = sum(acc for _, acc, _ in scores) / len(scores)
    mean_auc = float(np.nanmean([auc for _, _, auc in scores]))
    return mean_acc, mean_auc, max_batches is None


def save_best_models(models, config, logger, epoch_index, accuracy, auc, full_eval):
    """Overwrite the running best checkpoint for this training run."""
    root = getattr(config, 'MODEL_STORAGE_MAIN_ROOT', './Model_Storage_main')
    base_path = (
        f'{root}/{config.PARTICIPANT_PARAMS["loss_function"]}/'
        f'{config.NOISE_TYPE}{config.NOISE_RATE}/'
    )
    save_best_checkpoints(
        models, config.PRIVATE_NETS_NAME_LIST, base_path,
        epoch_index, accuracy, auc, full_eval, logger,
    )

def get_cifar10_dataloader(datadir, train_bs, test_bs, dataidxs=None, train_flag=True):
    """Get CIFAR-10 dataloader"""
    transform_train, transform_test = get_cifar10_transforms()
    
    if train_flag:
        transform = transform_train
    else:
        transform = transform_test
    
    dataset = CIFAR10Dataset(
        root=datadir,
        dataidxs=dataidxs,
        train=train_flag,
        transform=transform,
        target_transform=None,
        noise_type=None,
        noise_rate=0
    )
    
    batch_size = train_bs if train_flag else test_bs
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=train_flag,
        num_workers=2,
        pin_memory=True
    )
    
    return dataloader, dataset

def _keep_models_on_gpu():
    return bool(getattr(Config, 'KEEP_MODELS_ON_GPU', True)) and torch.cuda.is_available()


def stash_model(model, device):
    """After a train/eval step: keep on GPU (fast) or park to CPU (VRAM-saving)."""
    if _keep_models_on_gpu() and device.type == 'cuda':
        return model
    return model.cpu()


def models_to_device(models, device):
    for i in range(len(models)):
        models[i] = models[i].to(device)
    return models


def park_models_cpu(models):
    """Park all client nets on CPU (e.g. before Ollama takes the GPU)."""
    for i in range(len(models)):
        models[i] = models[i].cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def clear_gpu(model):
    """Release a temporary model reference; prefer stash_model for client nets."""
    del model
    if not _keep_models_on_gpu() and torch.cuda.is_available():
        torch.cuda.empty_cache()


def clear_image(images, labels):
    """Clear image and label tensors from GPU"""
    del images, labels


def clear_criterion(criterion):
    """Clear criterion from GPU"""
    del criterion

def _ckpt_dir(config):
    root = getattr(config, 'MODEL_STORAGE_MAIN_ROOT', './Model_Storage_main')
    return (
        f'{root}/{config.PARTICIPANT_PARAMS["loss_function"]}/'
        f'{config.NOISE_TYPE}{config.NOISE_RATE}'
    )


def _cpu_state_dict(model):
    raw = model.module if isinstance(model, nn.DataParallel) else model
    return {k: v.detach().cpu() for k, v in raw.state_dict().items()}


def _atomic_torch_save(obj, path):
    mkdirs(os.path.dirname(path))
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_party_checkpoint(model, path):
    """Atomically write one party's weights."""
    _atomic_torch_save(_cpu_state_dict(model), path)


def partial_party_path(config, index):
    netname = config.PRIVATE_NETS_NAME_LIST[index]
    return os.path.join(_ckpt_dir(config), 'partial', f'{netname}_{index}.ckpt')


def clear_partial_checkpoints(config):
    partial_dir = os.path.join(_ckpt_dir(config), 'partial')
    if not os.path.isdir(partial_dir):
        return
    for name in os.listdir(partial_dir):
        os.remove(os.path.join(partial_dir, name))


def _cpu_optimizer_state(optimizer):
    saved = optimizer.state_dict()
    state = {}
    for key, slot in saved.get('state', {}).items():
        state[key] = {
            name: value.detach().cpu() if torch.is_tensor(value) else value
            for name, value in slot.items()
        }
    return {'state': state, 'param_groups': saved.get('param_groups', [])}


def save_training_state(config, logger, epoch_index, stage, weight_quality,
                        weight_optimizer, mean_loss_list, selection_counts,
                        selected_clients, optimal_coefficients,
                        kl_done=None, gsam_done=None, gsam_party=None,
                        gsam_epochs_done=0):
    """Persist w_k and how far the current round got, so a kill can continue."""
    payload = {
        'version': 1,
        'epoch_index': int(epoch_index),
        'stage': stage,
        'kl_done': [int(p) for p in (kl_done or [])],
        'gsam_done': [int(p) for p in (gsam_done or [])],
        'gsam_party': None if gsam_party is None else int(gsam_party),
        'gsam_epochs_done': int(gsam_epochs_done),
        'selected_clients': [int(c) for c in (selected_clients or [])],
        'optimal_coefficients': [float(c) for c in (optimal_coefficients or [])],
        'mean_loss_list': [float(x) for x in (mean_loss_list or [])],
        'selection_counts': [int(c) for c in (selection_counts or [])],
        'weight_quality': weight_quality.detach().cpu(),
        'weight_optimizer': _cpu_optimizer_state(weight_optimizer),
    }
    path = os.path.join(_ckpt_dir(config), 'resume_state.pt')
    _atomic_torch_save(payload, path)
    logger.info(
        f'Saved resume state: Communication Epoch {int(epoch_index) + 1} '
        f'stage={stage} kl_done={payload["kl_done"]} '
        f'gsam_done={payload["gsam_done"]} '
        f'gsam_party={payload["gsam_party"]} '
        f'gsam_epochs_done={payload["gsam_epochs_done"]}'
    )


def load_training_state(config):
    path = os.path.join(_ckpt_dir(config), 'resume_state.pt')
    if not os.path.isfile(path):
        return None
    try:
        state = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        state = torch.load(path, map_location='cpu')
    if not isinstance(state, dict) or int(state.get('version', 0)) != 1:
        return None
    return state


def apply_training_state(state, weight_quality, optimizer):
    with torch.no_grad():
        saved = state['weight_quality'].to(device=weight_quality.device, dtype=weight_quality.dtype)
        weight_quality.data.copy_(saved)
    opt_state = state.get('weight_optimizer')
    if not opt_state:
        return
    optimizer.load_state_dict(opt_state)
    for slot in optimizer.state.values():
        for name, value in list(slot.items()):
            if torch.is_tensor(value):
                slot[name] = value.to(weight_quality.device)


def save_models_and_metrics(models, config, logger, epoch_index):
    """Save models and training metrics"""
    base_path = _ckpt_dir(config) + '/'
    mkdirs(base_path)
    
    for i, model in enumerate(models):
        netname = config.PRIVATE_NETS_NAME_LIST[i]
        save_party_checkpoint(model, f'{base_path}{netname}_{i}_epoch_{epoch_index}.ckpt')
    
    logger.info(f"Models and metrics saved successfully for epoch {epoch_index}")

def main():
    # Initialize logging
    logger = init_logs()
    metrics_logger = MetricsLogger(run_id=getattr(logger, 'run_id', None))
    logger.info("Initializing training configuration")
    logger.info(f'Metrics CSV: {metrics_logger.path}')
    log_run_config(logger, Config)
    
    # Set random seeds
    random.seed(Config.SEED)
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)
    
    # Setup device — GPU-first for all train/eval work.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if getattr(Config, 'REQUIRE_CUDA', True) and not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA is required for training (Config.REQUIRE_CUDA=True) but no GPU was found.'
        )
    # Timeshare: LLM may briefly borrow GPU, then unload. Training owns the rest.
    try:
        if getattr(Config, 'OLLAMA_ON_GPU', False):
            os.environ['OLLAMA_NUM_GPU'] = '999'
            logger.info('OLLAMA_ON_GPU=True: LLM borrows GPU for query windows, then unloads')
        else:
            os.environ['OLLAMA_NUM_GPU'] = '0'
            logger.info('OLLAMA_ON_GPU=False: LLM stays on CPU; GPU reserved for training')
        OllamaLLMClient().unload_model()
        logger.info('Cleared any previously loaded Ollama model from memory')
    except Exception as e:
        logger.warning(f'Could not reset Ollama at startup: {e}')
    device_ids = [0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_bench = bool(getattr(Config, 'CUDNN_BENCHMARK', True)) and device.type == 'cuda'
    torch.backends.cudnn.benchmark = use_bench
    torch.backends.cudnn.deterministic = not use_bench
    logger.info(
        f"Device={device}, KEEP_MODELS_ON_GPU={getattr(Config, 'KEEP_MODELS_ON_GPU', True)}, "
        f"CUDNN_BENCHMARK={use_bench}, PARALLEL_CLIENTS={Config.PARALLEL_CLIENTS}, "
        f"PARALLEL_EVAL={Config.PARALLEL_EVAL}, OLLAMA_ON_GPU={Config.OLLAMA_ON_GPU}, "
        f"OLLAMA_UNLOAD_AFTER_QUERY={getattr(Config, 'OLLAMA_UNLOAD_AFTER_QUERY', True)}"
    )
    if device.type == 'cuda':
        free_mb, total_mb = [x / (1024 ** 2) for x in torch.cuda.mem_get_info()]
        logger.info(f'GPU memory at start: {free_mb:.0f}/{total_mb:.0f} MiB free')
    
    # Initialize models and data
    n_parties = int(getattr(Config, 'N_PARTIES', Config.N_CLIENTS))
    client_ids = list(getattr(Config, 'CLIENT_IDS', list(range(Config.N_CLIENTS))))
    teacher_id = int(getattr(Config, 'SERVER_TEACHER_ID', 0))
    logger.info(
        f"Initializing models/data for {Config.DATASET_TYPE}: "
        f"N_PARTIES={n_parties}, N_CLIENTS={Config.N_CLIENTS}, "
        f"HAS_SERVER_TEACHER={getattr(Config, 'HAS_SERVER_TEACHER', False)}, "
        f"teacher={teacher_id}, CLIENT_IDS={client_ids}"
    )
    if Config.INTRA_DATASET:
        net_dataidx_map = {}
        for index in client_ids:
            idxes = np.random.permutation(50000)
            idxes = idxes[0:Config.PRIVATE_DATA_LEN]
            net_dataidx_map[index] = idxes
        logger.info(net_dataidx_map)
        net_list = init_nets(n_parties=n_parties,
                           nets_name_list=Config.NETS_NAME_LIST,
                           num_classes=Config.OUTPUT_CHANNEL)
    else:
        if Config.DATASET_TYPE == 'cifar10':
            (
                public_train_dataidx,
                public_test_dataidx,
                net_dataidx_map,
                test_dataidx_map,
            ) = split_cifar10_indices(
                n_clients=Config.N_CLIENTS,
                seed=Config.SEED,
                public_train=int(getattr(Config, 'CIFAR_PUBLIC_TRAIN', 5000)),
                public_test=int(getattr(Config, 'CIFAR_PUBLIC_TEST', 1000)),
                client_ids=client_ids,
            )
            logger.info(f"CIFAR-10 data split:")
            logger.info(f"  - Public train: {len(public_train_dataidx)} samples")
            logger.info(f"  - Public test: {len(public_test_dataidx)} samples")
            logger.info(f"  - Local train: {[(k, len(v)) for k, v in net_dataidx_map.items()]}")
            logger.info(f"  - Local test: {[(k, len(v)) for k, v in test_dataidx_map.items()]}")
            logger.info(f"Random seed used: {Config.SEED}")
        else:
            net_dataidx_map = {index: None for index in client_ids}
            logger.info(net_dataidx_map)
        
        net_list = init_nets(n_parties=n_parties,
                           nets_name_list=Config.NETS_NAME_LIST,
                           num_classes=Config.OUTPUT_CHANNEL)
    
    # Load all parties (0=server teacher on genimage_pub when HAS_SERVER_TEACHER).
    # resume_state.pt is authoritative: it stores every party's weights plus w_k
    # and the in-round stage. SMARTFED_RESUME_EPOCH is the fallback when that
    # file does not exist yet (last finished epoch_*.ckpt only).
    train_state = load_training_state(Config)
    resume_mid = None
    resume_from = 0
    if train_state is not None:
        if train_state.get('stage') == 'round_done':
            resume_from = int(train_state['epoch_index']) + 1
        else:
            resume_from = int(train_state['epoch_index'])
            resume_mid = train_state
        logger.info(
            f'Resume file: stage={train_state.get("stage")} '
            f'epoch_index={train_state.get("epoch_index")} '
            f'next Communication Epoch {resume_from + 1}'
        )
    else:
        resume_raw = os.environ.get('SMARTFED_RESUME_EPOCH', '').strip()
        resume_from = int(resume_raw) if resume_raw else 0
    # A stale resume_state.pt must not rewind a run that already saved later rounds.
    highest = -1
    for epoch in range(int(Config.COMMUNICATION_EPOCH)):
        complete = True
        for i in range(n_parties):
            netname = Config.PRIVATE_NETS_NAME_LIST[i]
            if not os.path.isfile(f'{_ckpt_dir(Config)}/{netname}_{i}_epoch_{epoch}.ckpt'):
                complete = False
                break
        if complete:
            highest = epoch
        else:
            break
    if highest + 1 > resume_from and (
        resume_mid is None or highest >= int(resume_mid['epoch_index'])
    ):
        logger.info(
            f'Epoch checkpoints reach epoch_{highest}; '
            f'continuing at Communication Epoch {highest + 2}/{Config.COMMUNICATION_EPOCH}'
        )
        train_state = None
        resume_mid = None
        resume_from = highest + 1
    if resume_from < 0 or resume_from > int(Config.COMMUNICATION_EPOCH):
        raise ValueError(
            f'Resume epoch {resume_from} is outside '
            f'0..{int(Config.COMMUNICATION_EPOCH)}'
        )
    if resume_from == int(Config.COMMUNICATION_EPOCH):
        logger.info('Resume state is already past the last round; nothing to run')
    base_epoch = resume_from - 1
    if resume_mid is not None:
        base_epoch = int(resume_mid['epoch_index']) - 1
    pretrained_model = []
    for i in range(n_parties):
        model_s0 = net_list[i]
        netname = Config.PRIVATE_NETS_NAME_LIST[i]
        if base_epoch >= 0:
            ckpt_path = (
                f'{_ckpt_dir(Config)}/{netname}_{i}_epoch_{base_epoch}.ckpt'
            )
        else:
            storage_root = getattr(Config, 'MODEL_STORAGE_ROOT', './Model_Storage')
            ckpt_path = (
                f'{storage_root}/{Config.PARTICIPANT_PARAMS["loss_function"]}/'
                f'{Config.NOISE_TYPE}{Config.NOISE_RATE}/{netname}_{i}.ckpt'
            )
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f'Resume checkpoint missing: {ckpt_path}')
        load_checkpoint(model_s0, ckpt_path, map_location='cpu')
        if resume_mid is not None:
            keep_partial = set(int(p) for p in (resume_mid.get('kl_done') or []))
            keep_partial.update(int(p) for p in (resume_mid.get('gsam_done') or []))
            if resume_mid.get('gsam_party') is not None:
                keep_partial.add(int(resume_mid['gsam_party']))
            partial_path = partial_party_path(Config, i)
            if i in keep_partial and os.path.isfile(partial_path):
                load_checkpoint(model_s0, partial_path, map_location='cpu')
                ckpt_path = partial_path
        role = 'server_teacher' if Config.is_server_teacher(i) else 'client'
        ds_key = Config.party_dataset_key(i)
        if _keep_models_on_gpu():
            model_s0 = model_s0.to(device)
            logger.info(f'Loaded party {i} ({role}/{ds_key}) on GPU from {ckpt_path}')
        else:
            logger.info(f'Loaded party {i} ({role}/{ds_key}) on CPU from {ckpt_path}')
        pretrained_model.append(model_s0)
    if device.type == 'cuda':
        free_mb = torch.cuda.mem_get_info()[0] / (1024 ** 2)
        logger.info(f'GPU free after model load: {free_mb:.0f} MiB')

    # CIFAR: distill from public_model.ckpt instead of client 0.
    kl_teacher_model = None
    if getattr(Config, 'PUBLIC_KL_USE_PUBLIC_MODEL', False):
        public_ckpt = (
            f'{getattr(Config, "MODEL_STORAGE_ROOT", "./Model_Storage")}/'
            f'{Config.PARTICIPANT_PARAMS["loss_function"]}/'
            f'{Config.NOISE_TYPE}{Config.NOISE_RATE}/public_model.ckpt'
        )
        if os.path.isfile(public_ckpt):
            kl_teacher_model = init_nets(
                n_parties=1,
                nets_name_list=[Config.NETS_NAME_LIST[0]],
                num_classes=Config.OUTPUT_CHANNEL,
            )[0]
            load_checkpoint(kl_teacher_model, public_ckpt, map_location='cpu')
            if _keep_models_on_gpu():
                kl_teacher_model = kl_teacher_model.to(device)
            logger.info(f'KL teacher loaded from public model: {public_ckpt}')
        else:
            logger.warning(
                f'PUBLIC_KL_USE_PUBLIC_MODEL=True but missing {public_ckpt}; '
                f'falling back to party {teacher_id}'
            )

    def _park_all_for_ollama():
        nonlocal kl_teacher_model
        park_models_cpu(pretrained_model)
        if kl_teacher_model is not None:
            kl_teacher_model = kl_teacher_model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _restore_all_after_ollama():
        nonlocal kl_teacher_model
        models_to_device(pretrained_model, device)
        if kl_teacher_model is not None:
            kl_teacher_model = kl_teacher_model.to(device)
    
    # Initialize public data loaders
    if Config.PUBLIC_DATASET_NAME == 'cifar10':
        # Use public data indices if available
        public_train_dataidx_to_use = public_train_dataidx if 'public_train_dataidx' in locals() else None
        public_test_dataidx_to_use = public_test_dataidx if 'public_test_dataidx' in locals() else None
        
        public_train_dl, public_train_ds = get_cifar10_dataloader(datadir=Config.PUBLIC_DATASET_DIR,
                                                                 train_bs=Config.TRAIN_BATCH_SIZE,
                                                                 test_bs=Config.TEST_BATCH_SIZE,
                                                                 dataidxs=public_train_dataidx_to_use,
                                                                 train_flag=True)
        public_test_dl, public_test_ds = get_cifar10_dataloader(datadir=Config.PUBLIC_DATASET_DIR,
                                                               train_bs=Config.TRAIN_BATCH_SIZE,
                                                               test_bs=Config.TEST_BATCH_SIZE,
                                                               dataidxs=public_test_dataidx_to_use,
                                                               train_flag=False)
    else:
        public_train_dl, public_train_ds = get_dataloader(dataset=Config.PUBLIC_DATASET_NAME,
                                                         datadir=Config.PUBLIC_DATASET_DIR,
                                                         train_bs=Config.TRAIN_BATCH_SIZE,
                                                         test_bs=Config.TEST_BATCH_SIZE,
                                                         dataidxs=None,
                                                         noise_type=Config.NOISE_TYPE,
                                                         noise_rate=Config.NOISE_RATE,
                                                         train_flag=True)
        
        public_test_dl, public_test_ds = get_dataloader(dataset=Config.PUBLIC_DATASET_NAME,
                                                       datadir=Config.PUBLIC_DATASET_DIR,
                                                       train_bs=Config.TRAIN_BATCH_SIZE,
                                                       test_bs=Config.TEST_BATCH_SIZE,
                                                       dataidxs=None,
                                                       noise_type=Config.NOISE_TYPE,
                                                       noise_rate=Config.NOISE_RATE,
                                                       train_flag=False)

    # Build private loaders once (path listing is cheap; avoid re-scanning every epoch).
    # Eval loaders use EVAL_BATCH_SIZE (larger OK: inference-only, no activations for backward).
    eval_bs = int(getattr(Config, 'EVAL_BATCH_SIZE', Config.TEST_BATCH_SIZE_PRIVATE))
    logger.info(
        f'Building private train/test DataLoaders (one-time); '
        f'eval_batch_size={eval_bs}, eval_amp={getattr(Config, "EVAL_USE_AMP", False)}'
    )
    private_train_dls = {}
    private_test_dls = {}
    for c_index in client_ids:
        private_dataidx = net_dataidx_map.get(c_index)
        if Config.PRIVATE_DATASET_NAME in ('genimage', 'diffusionface'):
            root_genim_full = Config.party_datadir(c_index)
            check_genimage_client_val(root_genim_full, logger=logger)
            train_dl_local, _ = get_dataloader(
                dataset=Config.PRIVATE_DATASET_NAME,
                datadir=root_genim_full,
                train_bs=Config.TRAIN_BATCH_SIZE_PRIVATE,
                test_bs=Config.TEST_BATCH_SIZE_PRIVATE,
                dataidxs=private_dataidx,
                noise_type=Config.NOISE_TYPE,
                noise_rate=Config.NOISE_RATE,
                train_flag=True,
            )
            test_dl_local, _ = get_dataloader(
                dataset=Config.PRIVATE_DATASET_NAME,
                datadir=root_genim_full,
                train_bs=eval_bs,
                test_bs=eval_bs,
                dataidxs=None,
                noise_type=Config.NOISE_TYPE,
                noise_rate=Config.NOISE_RATE,
                train_flag=False,
            )
        elif Config.PRIVATE_DATASET_NAME == 'cifar10':
            train_dl_local, _ = get_cifar10_dataloader(
                datadir=Config.PRIVATE_DATASET_DIR,
                train_bs=Config.TRAIN_BATCH_SIZE_PRIVATE,
                test_bs=Config.TEST_BATCH_SIZE_PRIVATE,
                dataidxs=private_dataidx,
                train_flag=True,
            )
            test_dataidx = test_dataidx_map[c_index] if 'test_dataidx_map' in locals() and test_dataidx_map is not None else None
            test_dl_local, _ = get_cifar10_dataloader(
                datadir=Config.PRIVATE_DATASET_DIR,
                train_bs=eval_bs,
                test_bs=eval_bs,
                dataidxs=test_dataidx,
                train_flag=False,
            )
        else:
            train_dl_local, _ = get_dataloader(
                dataset=Config.PRIVATE_DATASET_NAME,
                datadir=Config.PRIVATE_DATASET_DIR,
                train_bs=Config.TRAIN_BATCH_SIZE_PRIVATE,
                test_bs=Config.TEST_BATCH_SIZE_PRIVATE,
                dataidxs=private_dataidx,
                noise_type=Config.NOISE_TYPE,
                noise_rate=Config.NOISE_RATE,
                train_flag=True,
            )
            test_dl_local, _ = get_dataloader(
                dataset=Config.PRIVATE_DATASET_NAME,
                datadir=Config.PRIVATE_DATASET_DIR,
                train_bs=eval_bs,
                test_bs=eval_bs,
                dataidxs=None,
                noise_type=Config.NOISE_TYPE,
                noise_rate=Config.NOISE_RATE,
                train_flag=False,
            )
        private_train_dls[c_index] = train_dl_local
        private_test_dls[c_index] = test_dl_local
        logger.info(
            f'Client party {c_index} ({Config.party_dataset_key(c_index)}): '
            f'train={len(train_dl_local.dataset)}, test={len(test_dl_local.dataset)}'
        )

    if getattr(Config, 'DATA_CACHE_ENABLED', False):
        logger.info(
            f'SSD image cache ENABLED → {Config.DATA_CACHE_DIR} '
            f'(size={getattr(Config, "DATA_CACHE_IMAGE_SIZE", 224)}; '
            f'first read fills cache, later reads hit NVMe)'
        )
        os.makedirs(Config.DATA_CACHE_DIR, exist_ok=True)
        if getattr(Config, 'DATA_CACHE_PREWARM', False):
            from Dataset.image_cache import prewarm_paths
            logger.info('DATA_CACHE_PREWARM=True: populating cache (can take a while on first run)')
            for c_index in client_ids:
                for dl in (private_train_dls[c_index], private_test_dls[c_index]):
                    ok, fail = prewarm_paths(dl.dataset.data, logger=logger)
                    logger.info(f'Prewarm client {c_index}: ok={ok}, fail={fail}')
    
    # Main training loop
    col_loss_list = []
    local_loss_list = []
    acc_list = []
    current_mean_loss_list = []     
    
    # Initialize weight_quality as learnable parameters
    weight_quality = torch.full((Config.N_CLIENTS,), 1.0/Config.N_CLIENTS, requires_grad=True, device=device)
    weight_quality_optimizer = optim.Adam([weight_quality], lr=0.01)  # Learning rate for weight updates
    if train_state is not None and 'weight_quality' in train_state:
        apply_training_state(train_state, weight_quality, weight_quality_optimizer)
        if train_state.get('stage') == 'round_done' and train_state.get('mean_loss_list'):
            current_mean_loss_list = [float(x) for x in train_state['mean_loss_list']]
        logger.info(
            f'Restored w_k: {weight_quality.detach().cpu().numpy()}'
        )
    weight_rule = str(getattr(Config, 'WEIGHT_RULE', 'signed_loss')).lower()
    dataset_size_weights = None
    if weight_rule == 'dataset_size':
        size_counts = [
            len(private_train_dls[party_id].dataset) for party_id in client_ids
        ]
        size_total = float(sum(size_counts))
        if size_total <= 0:
            raise RuntimeError('Dataset-size weights need a positive train-set size')
        dataset_size_weights = [c / size_total for c in size_counts]
        logger.info(
            f'Dataset-size w_k counts={size_counts} weights='
            f'{[round(w, 6) for w in dataset_size_weights]}'
        )
    logger.info(
        f'KL weight rule (w_k): {weight_rule} '
        f'(uniform=1/K, dataset_size=n_k/sum n, signed_loss=APR delta-loss)'
    )
    
    # ---- Agent modes (R3-Major-4): W/o MA | Rule-based Eq.26 | LLM-driven ----
    dcsa_mode = getattr(Config, 'DCSA_MODE', 'llm')
    afwa_mode = getattr(Config, 'AFWA_MODE', 'llm')
    selection_budget = int(getattr(Config, 'SELECTION_BUDGET', Config.N_CLIENTS) or Config.N_CLIENTS)
    ollama_model = getattr(Config, 'OLLAMA_MODEL_NAME', 'llama3:8b')
    logger.info(
        f"Agent modes: AGENT_MODE={getattr(Config, 'AGENT_MODE', 'llm')}, "
        f"DCSA_MODE={dcsa_mode}, AFWA_MODE={afwa_mode}, "
        f"SELECTION_BUDGET={selection_budget}/{Config.N_CLIENTS}, "
        f"WEIGHT_RULE={getattr(Config, 'WEIGHT_RULE', 'signed_loss')}"
    )

    llm_selector = None
    if dcsa_mode == 'off':
        logger.info("DCSA disabled (W/o MA): all clients participate each round")
    elif dcsa_mode == 'rule_based':
        rule_strategy = getattr(Config, 'DCSA_RULE_STRATEGY', 'adaptive')
        llm_selector = create_llm_client_selector(
            n_clients=Config.N_CLIENTS,
            strategy=rule_strategy,
            logger=logger,
        )
        logger.info(f"DCSA rule-based selector initialized (strategy={rule_strategy}, no LLM)")
    else:
        llm_selector = create_llm_client_selector(
            n_clients=Config.N_CLIENTS,
            strategy="llama3",
            logger=logger,
        )
        logger.info(f"DCSA LLM selector initialized ({ollama_model})")

    loss_weighting_agent = None
    if afwa_mode == 'off':
        logger.info(
            f"AFWA disabled (W/o MA): fixed coefficient="
            f"{getattr(Config, 'AFWA_FIXED_COEFFICIENT', 0.1)}"
        )
    elif afwa_mode == 'rule_based':
        loss_weighting_agent = create_loss_weighting_agent(
            model_name=ollama_model,
            logger=logger,
            use_llm=False,
        )
        logger.info("AFWA rule-based Eq.(26) initialized (use_llm=False, no LLM calls)")
    else:
        loss_weighting_agent = create_loss_weighting_agent(
            model_name=ollama_model,
            logger=logger,
            use_llm=True,
        )
        logger.info(f"AFWA LLM agent initialized ({ollama_model})")

    GLOBAL_LLM_COST.reset_all()
    selection_counts = [0] * Config.N_CLIENTS
    if train_state is not None and train_state.get('selection_counts'):
        selection_counts = [int(c) for c in train_state['selection_counts']]
    best_across = {'acc': -1.0, 'auc': -1.0, 'round': None, 'full': False}
    if resume_from > 0 or resume_mid is not None:
        best_path = (
            f'{getattr(Config, "MODEL_STORAGE_MAIN_ROOT", "./Model_Storage_main")}/'
            f'{Config.PARTICIPANT_PARAMS["loss_function"]}/'
            f'{Config.NOISE_TYPE}{Config.NOISE_RATE}/best_metrics.json'
        )
        if os.path.isfile(best_path):
            with open(best_path, encoding='utf-8') as fh:
                meta = json.load(fh)
            best_auc = meta.get('auc')
            best_across = {
                'acc': float(meta['accuracy']),
                'auc': -1.0 if best_auc is None else float(best_auc),
                'round': int(meta['round']),
                'full': bool(meta.get('full_eval')),
            }
        logger.info(
            f'Resuming at Communication Epoch {resume_from + 1}/'
            f'{Config.COMMUNICATION_EPOCH}. '
            f'Kept best round {best_across["round"]} '
            f'Acc={best_across["acc"]:.2f}%, AUC={best_across["auc"]:.4f}'
        )

    for epoch_index in range(resume_from, Config.COMMUNICATION_EPOCH):
        logger.info(f"Communication Epoch {epoch_index + 1}/{Config.COMMUNICATION_EPOCH}")
        GLOBAL_LLM_COST.reset_round()

        mid = None
        if resume_mid is not None and int(resume_mid['epoch_index']) == epoch_index:
            mid = resume_mid
            resume_mid = None
        skip_to_kl = mid is not None and mid.get('stage') == 'kl'
        skip_to_eval = mid is not None and mid.get('stage') == 'post_kl'
        skip_to_gsam = mid is not None and mid.get('stage') in ('post_eval', 'gsam')
        skip_prefix = skip_to_kl or skip_to_eval or skip_to_gsam
        
        # Star topology only: no client↔client pair weights.
        quality_list = []
        quality_exp = []
        delta_mean_loss_list = []
        if skip_prefix:
            last_mean_loss_list = []
            current_mean_loss_list = [float(x) for x in mid.get('mean_loss_list') or []]
            logger.info(
                f'Resuming inside Communication Epoch {epoch_index + 1} '
                f'at stage {mid.get("stage")} '
                f'kl_done={mid.get("kl_done")} gsam_done={mid.get("gsam_done")} '
                f'gsam_party={mid.get("gsam_party")} '
                f'gsam_epochs_done={mid.get("gsam_epochs_done")}'
            )
        else:
            last_mean_loss_list = current_mean_loss_list
            current_mean_loss_list = []
            clear_partial_checkpoints(Config)
        
        # Client selection (DCSA).
        # W/o MA + full budget → all clients. Fig.4 (budget < N, DCSA=off) → random subset.
        if skip_prefix:
            selected_clients = [int(c) for c in mid.get('selected_clients') or []]
        elif dcsa_mode == 'off' or llm_selector is None:
            if selection_budget < Config.N_CLIENTS:
                rng = random.Random(int(Config.SEED) + epoch_index)
                selected_clients = sorted(
                    rng.sample(list(range(Config.N_CLIENTS)), selection_budget)
                )
            else:
                selected_clients = list(range(Config.N_CLIENTS))
        else:
            needs_ollama = dcsa_mode == 'llm'
            if needs_ollama and getattr(Config, 'OLLAMA_ON_GPU', False):
                _park_all_for_ollama()
            ollama_client = getattr(
                getattr(llm_selector, 'selection_strategy', None), 'ollama_client', None
            )
            with ollama_inference_session(
                ollama_client if needs_ollama else None,
                logger=logger,
            ):
                selected_clients = llm_selector.select_clients(
                    current_round=epoch_index,
                    selection_budget=selection_budget,
                    global_performance=0.8,
                )
            if needs_ollama and _keep_models_on_gpu():
                _restore_all_after_ollama()
            selected_clients = [int(c) for c in selected_clients]
            # Keep coverage: inject a never-selected client so LLM cannot starve parties.
            if selection_budget < Config.N_CLIENTS and epoch_index > 0 and selected_clients:
                never = [i for i in range(Config.N_CLIENTS) if selection_counts[i] == 0]
                if never:
                    pick = never[epoch_index % len(never)]
                    if pick not in selected_clients:
                        selected_clients[-1] = pick
        selected_clients = sorted({int(c) for c in selected_clients})
        if not skip_prefix:
            for c in selected_clients:
                if 0 <= c < Config.N_CLIENTS:
                    selection_counts[c] += 1
        logger.info(
            f"Selected clients for epoch {epoch_index} (DCSA={dcsa_mode}): {selected_clients}"
        )
        
        # Local-weight probe on private clients only (not server teacher).
        # LLM/AFWA use local indices 0..N_CLIENTS-1; models use party ids in client_ids.
        if not skip_prefix:
          for local_i, party_id in enumerate(client_ids):
            model_s1 = pretrained_model[party_id].to(device)
            model_s1.train()
            train_dl_local = private_train_dls[party_id]
            
            criterion = nn.CrossEntropyLoss()
            criterion.to(device)
            participant_loss_list = []
            local_weight_batches = int(getattr(Config, 'LOCAL_WEIGHT_MAX_BATCHES', 64))
            
            for batch_idx, (images_s1, labels_s1) in enumerate(train_dl_local):
                if batch_idx >= local_weight_batches:
                    break
                images_s1 = images_s1.to(device, non_blocking=True)
                labels_s1 = labels_s1.to(device, non_blocking=True)
                private_linear_output = model_s1(images_s1)
                private_loss = criterion(private_linear_output, labels_s1)
                participant_loss_list.append(private_loss.item())
            
            mean_participant_loss = mean(participant_loss_list) if participant_loss_list else 0.0
            current_mean_loss_list.append(mean_participant_loss)
            
            client_accuracy_pct, client_auc = compute_accuracy_and_auc(
                model_s1, train_dl_local, device, max_batches=local_weight_batches
            )
            client_accuracy = client_accuracy_pct / 100.0
            
            if llm_selector is not None:
                llm_selector.update_client_metrics(
                    client_id=local_i,
                    loss=mean_participant_loss,
                    accuracy=client_accuracy,
                    data_quality=1.0,
                    performance_trend=0.0,
                    resource_availability=1.0,
                    model_diversity=random.uniform(0.3, 0.8),
                )
            
            pretrained_model[party_id] = stash_model(model_s1, device)
            logger.info(
                f'Calculate local weight of party {party_id} '
                f'({Config.party_dataset_key(party_id)}), '
                f'Loss: {mean_participant_loss:.4f}, '
                f'Accuracy: {client_accuracy_pct:.2f}%, AUC: {client_auc:.4f}'
            )
            if metrics_logger is not None:
                metrics_logger.log(
                    phase='local_weight',
                    client_id=party_id,
                    round=epoch_index,
                    loss=mean_participant_loss,
                    accuracy=client_accuracy_pct,
                    auc=client_auc,
                )
            clear_image(images=images_s1, labels=labels_s1)
            clear_criterion(criterion=criterion)
        
        # w_k for the KL term. Only signed_loss runs the APR update; the other
        # rules keep KL and replace that update (R2-4 / Table VII).
        if skip_prefix:
            pass
        elif weight_rule == 'uniform':
            with torch.no_grad():
                weight_quality.data = torch.ones(Config.N_CLIENTS, device=device) / Config.N_CLIENTS
        elif weight_rule == 'dataset_size':
            with torch.no_grad():
                weight_quality.data = torch.tensor(
                    dataset_size_weights, device=device, dtype=weight_quality.dtype
                )
        elif epoch_index > 0 and len(last_mean_loss_list) == Config.N_CLIENTS:
            # Signed-loss APR: quality from the loss change, then a softmax step.
            for c_index in range(Config.N_CLIENTS):
                delta_loss = last_mean_loss_list[c_index] - current_mean_loss_list[c_index]
                quality_list.append(delta_loss / current_mean_loss_list[c_index])
            
            # Normalize quality list
            quality_sum = sum(quality_list)
            normalized_quality = [q / quality_sum for q in quality_list]
            
            # Update weight_quality using gradient descent
            weight_quality_optimizer.zero_grad()
            
            # Calculate loss for weight_quality update
            # We want to maximize the weighted sum of quality improvements
            quality_tensor = torch.tensor(normalized_quality, device=device)
            weight_loss = -torch.sum(weight_quality * quality_tensor)  # Negative because we want to maximize
            
            # Add regularization to keep weights balanced
            regularization = 0.1 * torch.sum((weight_quality - 1.0/Config.N_CLIENTS) ** 2)
            total_loss = weight_loss + regularization
            
            total_loss.backward()
            weight_quality_optimizer.step()
            
            # Normalize weights to sum to 1
            with torch.no_grad():
                weight_quality.data = torch.softmax(weight_quality.data, dim=0)
        else:
            # First epoch of signed-loss APR: uniform weights
            with torch.no_grad():
                weight_quality.data = torch.ones(Config.N_CLIENTS, device=device) / Config.N_CLIENTS
        
        logger.info(
            f'Local weights ({weight_rule}): {weight_quality.detach().cpu().numpy()}'
        )

        # AFWA once per communication round (NOT every public batch).
        fixed_coef = float(getattr(Config, 'AFWA_FIXED_COEFFICIENT', 0.1))
        if skip_prefix and mid.get('optimal_coefficients'):
            optimal_coefficients = [float(c) for c in mid['optimal_coefficients']]
            logger.info(
                f'Restored AFWA coefficients: {[round(c, 4) for c in optimal_coefficients]}'
            )
        elif afwa_mode == 'off' or loss_weighting_agent is None:
            optimal_coefficients = [fixed_coef] * Config.N_CLIENTS
            logger.info(f'AFWA off: fixed coefficients={[round(c, 4) for c in optimal_coefficients]}')
        else:
            global_loss = np.mean(current_mean_loss_list) if current_mean_loss_list else 0.0
            global_accuracy = 0.8
            client_losses = current_mean_loss_list if current_mean_loss_list else [0.5] * Config.N_CLIENTS
            client_accuracies = [0.8] * Config.N_CLIENTS
            client_weights = weight_quality.detach().cpu().numpy().tolist()
            weighting_context = LossWeightingContext(
                current_round=epoch_index,
                total_rounds=Config.COMMUNICATION_EPOCH,
                global_loss=global_loss,
                global_accuracy=global_accuracy,
                client_losses=client_losses,
                client_accuracies=client_accuracies,
                client_weights=client_weights,
                convergence_rate=0.1 if epoch_index > 0 else 0.0,
                diversity_score=0.7,
                previous_weighting_effectiveness=0.8 if epoch_index > 0 else 0.0,
                training_stability=0.9,
                communication_efficiency=0.85,
            )

            def query_coefficient(c_index):
                coef = loss_weighting_agent.calculate_loss_weighting(
                    weighting_context, c_index
                )
                logger.info(f"Client {c_index}: Loss weighting coefficient = {coef:.4f}")
                return coef

            ollama_workers = max(1, int(getattr(Config, 'OLLAMA_MAX_PARALLEL', 1)))
            use_afwa_llm = bool(getattr(loss_weighting_agent, 'use_llm', False))
            logger.info(
                f'Querying AFWA coefficients once this round '
                f'(mode={afwa_mode}, use_llm={use_afwa_llm}, '
                f'serial_workers={ollama_workers}, '
                f'timeout={getattr(Config, "OLLAMA_TIMEOUT", 30)}s, '
                f'on_gpu={Config.OLLAMA_ON_GPU})'
            )
            if use_afwa_llm and getattr(Config, 'OLLAMA_ON_GPU', False):
                _park_all_for_ollama()
            with ollama_inference_session(
                loss_weighting_agent.ollama_client if use_afwa_llm else None,
                logger=logger,
            ):
                optimal_coefficients = map_parallel(
                    query_coefficient,
                    list(range(Config.N_CLIENTS)),
                    max_workers=ollama_workers if use_afwa_llm else 1,
                )
            if use_afwa_llm and _keep_models_on_gpu():
                _restore_all_after_ollama()
            for coef in optimal_coefficients:
                loss_weighting_agent.update_effectiveness(0.1, coef, 0.01)
            logger.info(f'AFWA coefficients: {[round(c, 4) for c in optimal_coefficients]}')
        
        # Public KL — star topology: server teacher ↔ one client at a time.
        # Pairs: (0,1), (0,2), …, (0,N). No client↔client soft-labels or updates.
        if bool(getattr(Config, 'PUBLIC_KL_CONSENSUS', False)):
            logger.warning(
                'PUBLIC_KL_CONSENSUS=True is ignored: star topology requires '
                'soft targets from the server teacher only.'
            )
        public_kl_batches = int(getattr(Config, 'PUBLIC_KL_MAX_BATCHES', 20))
        public_kl_lr = float(getattr(Config, 'PUBLIC_KL_LR', 5e-5))
        kl_temp = float(getattr(Config, 'PUBLIC_KL_TEMPERATURE', 2.0))
        kl_head_only = bool(getattr(Config, 'PUBLIC_KL_HEAD_ONLY', True))
        freeze_teacher = bool(getattr(Config, 'PUBLIC_KL_FREEZE_TEACHER', True))
        logger.info(
            f'Public KL (star): teacher={teacher_id}↔each of {client_ids}, '
            f'batches/pair={public_kl_batches}, lr={public_kl_lr}, T={kl_temp}, '
            f'head_only={kl_head_only}, freeze_teacher={freeze_teacher}'
        )

        def _set_kl_trainable(model, head_only):
            """Freeze backbone during KL so private features are not wiped."""
            if not head_only:
                for p in model.parameters():
                    p.requires_grad = True
                return
            for p in model.parameters():
                p.requires_grad = False
            unfrozen = False
            for attr in ('classifier', 'fc', 'linear'):
                mod = getattr(model, attr, None)
                if mod is None:
                    continue
                for p in mod.parameters():
                    p.requires_grad = True
                    unfrozen = True
            if not unfrozen:
                last_linear = None
                for m in model.modules():
                    if isinstance(m, nn.Linear):
                        last_linear = m
                if last_linear is not None:
                    for p in last_linear.parameters():
                        p.requires_grad = True

        kl_criterion = nn.KLDivLoss(reduction='batchmean').to(device)
        if kl_teacher_model is not None:
            kl_teacher = kl_teacher_model.to(device)
            teacher_label = 'public_model'
        else:
            kl_teacher = pretrained_model[teacher_id].to(device)
            teacher_label = f'party {teacher_id}'
        if freeze_teacher or getattr(Config, 'PUBLIC_KL_SKIP_TEACHER0', True):
            for p in kl_teacher.parameters():
                p.requires_grad = False
        kl_teacher.eval()

        kl_student_ids = list(enumerate(client_ids))
        if selection_budget < Config.N_CLIENTS:
            selected_set = set(selected_clients)
            kl_student_ids = [
                (local_i, party_id)
                for local_i, party_id in kl_student_ids
                if local_i in selected_set
            ]
            logger.info(
                f'Public KL students this round (selected only): '
                f'{[p for _, p in kl_student_ids]}'
            )

        def _checkpoint(stage, kl_ids, gsam_ids_done, gsam_party, gsam_epochs_done):
            save_training_state(
                Config, logger, epoch_index, stage, weight_quality,
                weight_quality_optimizer, current_mean_loss_list, selection_counts,
                selected_clients, optimal_coefficients,
                kl_done=kl_ids, gsam_done=gsam_ids_done,
                gsam_party=gsam_party, gsam_epochs_done=gsam_epochs_done,
            )

        run_kl = not (skip_to_eval or skip_to_gsam)
        kl_done = set(int(p) for p in ((mid or {}).get('kl_done') or []))
        if run_kl:
            _checkpoint('kl', sorted(kl_done), [], None, 0)

        for local_i, party_id in kl_student_ids:
            if not run_kl or party_id in kl_done:
                if party_id in kl_done:
                    logger.info(f'Skip KL client {party_id}; weights already saved')
                continue
            if kl_teacher_model is None and party_id == teacher_id:
                logger.info(f'Skip KL on teacher party {party_id}')
                continue
            logger.info(
                f'Public KL pair: {teacher_label} ↔ client {party_id} '
                f'({Config.party_dataset_key(party_id)})'
            )
            student = pretrained_model[party_id].to(device)
            _set_kl_trainable(student, kl_head_only)
            kl_opt = optim.Adam(
                (p for p in student.parameters() if p.requires_grad),
                lr=public_kl_lr,
            )
            pair_losses = []

            for batch_idx, (images_s2, _) in enumerate(public_train_dl):
                if batch_idx >= public_kl_batches:
                    break
                if batch_idx % 10 == 0:
                    logger.info(
                        f'  KL batch {batch_idx}/{public_kl_batches} '
                        f'({teacher_label} ↔ client {party_id})'
                    )
                images_s2_gpu = images_s2.to(device, non_blocking=True)

                with torch.no_grad():
                    target = F.softmax(kl_teacher(images_s2_gpu) / kl_temp, dim=1)

                student.eval()  # BN aligned with teacher.eval()
                kl_opt.zero_grad(set_to_none=True)
                student_log = F.log_softmax(student(images_s2_gpu) / kl_temp, dim=1)
                w = weight_quality[local_i].detach()
                loss = (
                    (kl_temp ** 2)
                    * optimal_coefficients[local_i]
                    * w
                    * kl_criterion(student_log, target)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (p for p in student.parameters() if p.requires_grad),
                    max_norm=1.0,
                )
                kl_opt.step()
                pair_losses.append(loss.item())
                del images_s2_gpu, target

            col_loss_list.append(pair_losses)
            for p in student.parameters():
                p.requires_grad = True
            pretrained_model[party_id] = stash_model(student, device)
            logger.info(
                f'Finished KL pair {teacher_label} ↔ client {party_id}: '
                f'mean_loss={mean(pair_losses) if pair_losses else 0.0:.6f}'
            )
            save_party_checkpoint(pretrained_model[party_id], partial_party_path(Config, party_id))
            kl_done.add(party_id)
            _checkpoint('kl', sorted(kl_done), [], None, 0)

        if run_kl:
            _checkpoint('post_kl', sorted(kl_done), [], None, 0)

        if kl_teacher_model is None:
            for p in pretrained_model[teacher_id].parameters():
                p.requires_grad = True
            pretrained_model[teacher_id] = stash_model(pretrained_model[teacher_id], device)
        else:
            kl_teacher_model = stash_model(kl_teacher, device)
        logger.info('Public KL (star) finished for all server↔client pairs')

        if skip_to_gsam:
            logger.info('Skip post-KL probe and across-eval; resume continues at GSAM')
        else:
            # Quick own-dataset probe (clients only).
            probe_batches = min(8, int(getattr(Config, 'EVAL_MAX_BATCHES', 64) or 8))
            for party_id in client_ids:
                model_probe = pretrained_model[party_id].to(device)
                acc_p, auc_p = compute_accuracy_and_auc(
                    model_probe, private_test_dls[party_id], device, max_batches=probe_batches
                )
                logger.info(
                    f'Post-KL own-val probe party {party_id} '
                    f'({Config.party_dataset_key(party_id)}): Acc={acc_p:.2f}%, AUC={auc_p:.4f}'
                )
                pretrained_model[party_id] = stash_model(model_probe, device)

            # Across-eval every round: subsample by default; full val every N rounds + last.
            eval_every = max(1, int(getattr(Config, 'EVAL_EVERY_N_ROUNDS', 5)))
            is_last_round = (epoch_index == Config.COMMUNICATION_EPOCH - 1)
            full_eval = is_last_round or (epoch_index % eval_every == 0)
            logger.info(
                f'Across-eval this round: full_eval={full_eval} '
                f'(every {eval_every} rounds + last; else EVAL_MAX_BATCHES='
                f'{getattr(Config, "EVAL_MAX_BATCHES", None)})'
            )
            mean_acc, mean_auc, eval_is_full = across_evaluation(
                pretrained_model, Config.N_CLIENTS, Config.PRIVATE_DATASET_NAME,
                Config.PRIVATE_DATASET_DIR, Config.TRAIN_BATCH_SIZE, Config.TEST_BATCH_SIZE,
                Config.NOISE_TYPE, Config.NOISE_RATE, logger, device, None,
                metrics_logger=metrics_logger, round_idx=epoch_index,
                cached_test_dls=private_test_dls,
                full_eval=full_eval,
            )
            if mean_acc is not None and across_is_better(
                mean_acc, mean_auc, eval_is_full, best_across
            ):
                best_across = {
                    'acc': mean_acc,
                    'auc': mean_auc if mean_auc == mean_auc else -1.0,
                    'round': epoch_index,
                    'full': bool(eval_is_full),
                }
                save_best_models(
                    pretrained_model, Config, logger, epoch_index,
                    mean_acc, mean_auc, eval_is_full,
                )
                if metrics_logger is not None:
                    metrics_logger.log(
                        phase='best_across_eval',
                        round=epoch_index,
                        accuracy=mean_acc,
                        auc=mean_auc,
                        extra=f'full_eval={bool(eval_is_full)}',
                    )
            metrics_logger.log(
                phase='round_local_loss',
                round=epoch_index,
                loss=float(np.mean(current_mean_loss_list)) if current_mean_loss_list else None,
            )
            _checkpoint('post_eval', sorted(kl_done), [], None, 0)

        # GSAM updates: clients always; server (party 0) only under Homo_ServerGSAM.
        # When DCSA is active, only selected clients run private GSAM (Fig.4 ablation).
        gsam_ids = Config.gsam_party_ids()
        if dcsa_mode != 'off':
            selected_party_ids = {
                client_ids[i] for i in selected_clients if 0 <= i < len(client_ids)
            }
            gsam_ids = [
                p for p in gsam_ids
                if Config.is_server_teacher(p) or p in selected_party_ids
            ]
        gsam_done = set(int(p) for p in ((mid or {}).get('gsam_done') or []))
        local_loss_batch_list = [None] * Config.N_CLIENTS
        server_gsam_loss = None
        logger.info(
            f'GSAM parties this round: {gsam_ids} '
            f'(GSAM_UPDATE_SERVER={getattr(Config, "GSAM_UPDATE_SERVER", False)}, '
            f'DCSA={dcsa_mode})'
        )

        # Train/eval loaders: clients → private_*; server → public_* (genimage_pub).
        gsam_train_dls = dict(private_train_dls)
        gsam_test_dls = dict(private_test_dls)
        if teacher_id in gsam_ids:
            gsam_train_dls[teacher_id] = public_train_dl
            gsam_test_dls[teacher_id] = public_test_dl

        def private_gsam_train(party_id):
            model_s4 = pretrained_model[party_id].to(device)
            model_s4.train()
            train_dl_local = gsam_train_dls[party_id]
            test_dl = gsam_test_dls[party_id]
            role = 'server_teacher' if Config.is_server_teacher(party_id) else 'client'
            gsam_epochs = int(getattr(Config, 'GSAM_TOTAL_EPOCHS', 10))
            start_epoch = 0
            if (
                mid is not None
                and mid.get('gsam_party') is not None
                and int(mid['gsam_party']) == int(party_id)
            ):
                start_epoch = int(mid.get('gsam_epochs_done') or 0)
            logger.info(
                f'GSAM start party {party_id} ({role}/{Config.party_dataset_key(party_id)}) '
                f'from local epoch {start_epoch + 1}/{gsam_epochs}'
            )
            if start_epoch >= gsam_epochs:
                return party_id, 0.0

            def on_gsam_epoch(model, epoch_done):
                save_party_checkpoint(model, partial_party_path(Config, party_id))
                _checkpoint(
                    'gsam', sorted(kl_done), sorted(gsam_done),
                    party_id, epoch_done + 1,
                )

            base_optimizer = optim.SGD(
                model_s4.parameters(),
                lr=float(getattr(Config, 'GSAM_LR', 0.01)),
            )
            criterion = nn.CrossEntropyLoss()
            gsam_optimizer = GSAMOptimizer(
                model_s4,
                base_optimizer,
                rho_max=float(getattr(Config, 'GSAM_RHO_MAX', 0.05)),
                rho_min=float(getattr(Config, 'GSAM_RHO_MIN', 0.01)),
                alpha=float(getattr(Config, 'GSAM_ALPHA', 0.5)),
                total_epochs=gsam_epochs,
            )
            model_s4, private_loss_batch_list = train_gsam(
                model=model_s4,
                train_dataloader=train_dl_local,
                eval_dataloader=test_dl,
                criterion=criterion,
                optimizer=gsam_optimizer,
                total_epochs=gsam_epochs,
                logger=logger,
                device=device,
                start_epoch=start_epoch,
                on_epoch_end=on_gsam_epoch,
            )
            mean_private_loss_batch = mean(private_loss_batch_list) if private_loss_batch_list else 0.0
            pretrained_model[party_id] = stash_model(model_s4, device)
            return party_id, mean_private_loss_batch

        for chunk in iter_chunks(gsam_ids, Config.PARALLEL_CLIENTS):
            pending = [party_id for party_id in chunk if party_id not in gsam_done]
            if not pending:
                logger.info(f'Skip finished GSAM parties: {list(chunk)}')
                continue
            logger.info(f'GSAM party chunk: {list(pending)}')
            results = map_parallel(private_gsam_train, pending, max_workers=len(pending))
            for party_id, mean_loss in results:
                if Config.is_server_teacher(party_id):
                    server_gsam_loss = mean_loss
                    logger.info(f'Server GSAM mean loss: {mean_loss:.6f}')
                    if metrics_logger is not None:
                        metrics_logger.log(
                            phase='server_gsam',
                            client_id=party_id,
                            round=epoch_index,
                            loss=float(mean_loss),
                        )
                else:
                    local_loss_batch_list[Config.client_local_index(party_id)] = mean_loss
                gsam_done.add(party_id)
                _checkpoint('gsam', sorted(kl_done), sorted(gsam_done), None, 0)
            if not _keep_models_on_gpu():
                torch.cuda.empty_cache()

        local_loss_list.append(local_loss_batch_list)
        if server_gsam_loss is not None:
            logger.info(f'Round {epoch_index}: server GSAM loss={server_gsam_loss:.6f}')
        
        # Train reward model and update LLM selector (LLM DCSA only).
        if llm_selector is not None and dcsa_mode == 'llm' and epoch_index > 0:
            client_metrics = []
            performance_feedback = []
            if llm_selector.client_metrics_history:
                for c_index in range(Config.N_CLIENTS):
                    if c_index in llm_selector.client_metrics_history[-1]:
                        metrics = llm_selector.client_metrics_history[-1][c_index]
                        client_metrics.append(metrics)
                        performance_feedback.append(metrics.accuracy)
            if client_metrics and performance_feedback:
                llm_selector.train_reward_model(client_metrics, performance_feedback)

        # Per-round LLM cost (Table VI): calls, latency, fallback rate.
        if getattr(Config, 'LOG_LLM_COST', True):
            round_cost = GLOBAL_LLM_COST.consume_round_summary()
            logger.info(
                f'LLM cost round {epoch_index}: calls={round_cost["calls"]}, '
                f'avg_latency_ms={round_cost["avg_latency_ms"]:.1f}, '
                f'fallback_rate={100.0 * round_cost["fallback_rate"]:.2f}%'
            )
            metrics_logger.log(
                phase='llm_cost',
                round=epoch_index,
                extra=json.dumps(round_cost),
            )
        
        # Get and log selection / AFWA statistics
        if epoch_index % 5 == 0:
            if llm_selector is not None:
                stats = llm_selector.get_selection_statistics()
                logger.info(f"Selection Statistics (Epoch {epoch_index}): {stats}")
            if loss_weighting_agent is not None:
                weighting_stats = loss_weighting_agent.get_weighting_statistics()
                logger.info(f"Loss Weighting Statistics (Epoch {epoch_index}): {weighting_stats}")
        
        # Save reward model periodically
        if llm_selector is not None and epoch_index % 10 == 0 and epoch_index > 0:
            main_root = getattr(Config, 'MODEL_STORAGE_MAIN_ROOT', './Model_Storage_main')
            mkdirs(main_root)
            reward_model_path = f'{main_root}/reward_model_epoch_{epoch_index}.pth'
            llm_selector.save_reward_model(reward_model_path)
        
        # Save models, w_k, and the loss list after each finished communication round.
        save_models_and_metrics(pretrained_model, Config, logger, epoch_index)
        _checkpoint('round_done', sorted(kl_done), sorted(gsam_done), None, 0)

    cum_cost = GLOBAL_LLM_COST.cumulative_summary()
    logger.info(f'LLM cost cumulative: {cum_cost}')
    metrics_logger.log(phase='llm_cost_cumulative', extra=json.dumps(cum_cost))
    if best_across.get('round') is not None:
        logger.info(
            f'Best across-eval kept: round {best_across["round"]} '
            f'Acc={best_across["acc"]:.2f}%, AUC={best_across["auc"]:.4f}, '
            f'full_eval={best_across["full"]}'
        )
    logger.info(f'Training finished. Text log: {logger.logfile}')
    logger.info(f'Metrics CSV: {metrics_logger.path}')
    metrics_logger.close()

if __name__ == '__main__':
    main()


