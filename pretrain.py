import os
import sys
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import Module

# Add project root to path
sys.path.append("..")
sys.path.append('./')

from Dataset.utils import (
    init_logs, get_dataloader, init_nets, mkdirs, compute_accuracy_and_auc,
    MetricsLogger, log_run_config, split_cifar10_indices,
)
from Dataset.integrity import check_genimage_client_val
from Dataset.init_dataset import genimage
from Dataset.cifar10_dataset import CIFAR10Dataset, get_cifar10_transforms
from Network.Models_Def.resnet import ResNet10, ResNet12
from loss import SCELoss
from config import Config
from parallel_utils import iter_chunks, map_parallel


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

def clear_gpu(model):
    """Clear GPU memory"""
    model = model.to('cpu')
    del model
    torch.cuda.empty_cache()

def setup_device(seed: int) -> torch.device:
    """Set up the device and random seeds for reproducibility.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        torch.device: The device to use for training
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if getattr(Config, 'REQUIRE_CUDA', True) and not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA is required for training (Config.REQUIRE_CUDA=True) but no GPU was found.'
        )
    # Pretrain does not need Ollama; keep GPU free for the nets.
    os.environ.setdefault('OLLAMA_NUM_GPU', '0')
    device_ids = [0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bench = bool(getattr(Config, 'CUDNN_BENCHMARK', True)) and device.type == 'cuda'
    torch.backends.cudnn.benchmark = use_bench
    torch.backends.cudnn.deterministic = not use_bench
    
    return device, device_ids

def pretrain_network(
    epoch: int,
    model: Module,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    loss_function: str,
    optimizer_name: str,
    learning_rate: float,
    num_classes: int,
    device: torch.device,
    logger: Any,
    metrics_logger: Optional[MetricsLogger] = None,
    client_id: Optional[int] = None,
) -> Module:
    """Pretrain a neural network.
    
    Args:
        epoch: Number of training epochs
        model: Neural network model
        train_dataloader: Training data loader
        eval_dataloader: Evaluation data loader
        loss_function: Name of loss function to use
        optimizer_name: Name of optimizer to use
        learning_rate: Learning rate for optimizer
        num_classes: Number of output classes
        device: Device to use for training
        logger: Logger for logging training progress
        metrics_logger: Optional CSV metrics writer
        client_id: Client index for metrics rows
        
    Returns:
        Module: Network restored to the best eval checkpoint (Acc, then AUC).
    """
    # Setup loss function
    if loss_function == 'CE':
        criterion = nn.CrossEntropyLoss()
    elif loss_function == 'SCE':
        criterion = SCELoss(alpha=0.1, beta=1, num_classes=num_classes)
    criterion.to(device)
    
    # Setup optimizer
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    best_acc = -1.0
    best_auc = float('nan')
    best_epoch = -1
    best_state = None
    
    # Training loop
    for _epoch in range(epoch):
        model.train()  # must re-enable train mode after each eval
        log_interval = 100
        running_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1
            
            # Logging
            if batch_idx % log_interval == 0:
                logger.info(
                    f'Train Epoch: {_epoch} [{batch_idx * len(images)}/{len(train_dataloader.dataset)} '
                    f'({100. * batch_idx / len(train_dataloader):.0f}%)]\tLoss: {loss.item():.6f}'
                )
                if metrics_logger is not None:
                    metrics_logger.log(
                        phase='pretrain_batch',
                        client_id=client_id,
                        epoch=_epoch,
                        loss=loss.item(),
                        extra=f'batch={batch_idx}',
                    )
        
        # Evaluation
        accuracy, auc = compute_accuracy_and_auc(model, eval_dataloader, device)
        avg_loss = running_loss / max(num_batches, 1)
        logger.info(
            f'Evaluate Accuracy: {accuracy:.2f}%, AUC: {auc:.4f}, '
            f'AvgTrainLoss: {avg_loss:.6f}'
        )
        if metrics_logger is not None:
            metrics_logger.log(
                phase='pretrain_epoch',
                client_id=client_id,
                epoch=_epoch,
                loss=avg_loss,
                accuracy=accuracy,
                auc=auc,
            )

        auc_cmp = auc if auc == auc else -1.0  # NaN → lose ties
        best_auc_cmp = best_auc if best_auc == best_auc else -1.0
        if accuracy > best_acc or (accuracy == best_acc and auc_cmp > best_auc_cmp):
            best_acc = accuracy
            best_auc = auc
            best_epoch = _epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            logger.info(
                f'New best checkpoint at epoch {_epoch}: '
                f'Acc={best_acc:.2f}%, AUC={best_auc:.4f}'
            )
            if metrics_logger is not None:
                metrics_logger.log(
                    phase='pretrain_best',
                    client_id=client_id,
                    epoch=_epoch,
                    accuracy=best_acc,
                    auc=best_auc,
                )

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(
            f'Restored best model (epoch {best_epoch}): '
            f'Acc={best_acc:.2f}%, AUC={best_auc:.4f}'
        )
    
    return model

def evaluate_network(
    model: Module,
    dataloader: DataLoader,
    device: torch.device,
    logger: Any
):
    """Evaluate a neural network (Accuracy + AUC).
    
    Args:
        model: Neural network model
        dataloader: Data loader for evaluation
        device: Device to use for evaluation
        logger: Logger for logging evaluation results
        
    Returns:
        Tuple[float, float]: Accuracy percentage and AUC
    """
    accuracy, auc = compute_accuracy_and_auc(model, dataloader, device)
    logger.info(f'Test Accuracy: {accuracy:.2f}%, AUC: {auc:.4f}')
    return accuracy, auc

def main():
    """Main training function."""
    # Setup logging and device
    mkdirs(f'{getattr(Config, "MODEL_STORAGE_ROOT", "./Model_Storage")}/{Config.PARTICIPANT_PARAMS["loss_function"]}/{Config.NOISE_TYPE}{Config.NOISE_RATE}')
    logger = init_logs()
    metrics_logger = MetricsLogger(run_id=getattr(logger, 'run_id', None))
    logger.info(f'Metrics CSV: {metrics_logger.path}')
    log_run_config(logger, Config)
    device, device_ids = setup_device(Config.SEED)
    
    logger.info(f"Loading Participants' Data and Model for {Config.DATASET_TYPE}")
    
    n_parties = int(getattr(Config, 'N_PARTIES', Config.N_CLIENTS))
    all_party_ids = list(range(n_parties))
    client_ids = list(getattr(Config, 'CLIENT_IDS', list(range(Config.N_CLIENTS))))
    pretrain_only = getattr(Config, 'PRETRAIN_PARTY_IDS', None)
    if pretrain_only:
        party_ids = [int(p) for p in pretrain_only]
        bad = [p for p in party_ids if p not in all_party_ids]
        if bad:
            raise ValueError(f'PRETRAIN_PARTY_IDS out of range: {bad}; valid={all_party_ids}')
    else:
        party_ids = list(all_party_ids)
    logger.info(
        f'Parties={n_parties}, HAS_SERVER_TEACHER={getattr(Config, "HAS_SERVER_TEACHER", False)}, '
        f'CLIENT_IDS={client_ids}, PRETRAIN_PARTY_IDS={party_ids}'
    )

    # Initialize networks (includes server teacher as party 0 when enabled)
    if Config.INTRA_DATASET:
        net_dataidx_map = {}
        for index in client_ids:
            idxes = np.random.permutation(50000)
            idxes = idxes[0:Config.PRIVATE_DATA_LEN]
            net_dataidx_map[index] = idxes
        logger.info(net_dataidx_map)
        net_list = init_nets(
            n_parties=n_parties,
            nets_name_list=Config.NETS_NAME_LIST,
            num_classes=Config.OUTPUT_CHANNEL
        )
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
            net_dataidx_map = {index: None for index in all_party_ids}
            logger.info(net_dataidx_map)
        
        net_list = init_nets(
            n_parties=n_parties,
            nets_name_list=Config.NETS_NAME_LIST,
            num_classes=Config.OUTPUT_CHANNEL
        )
    
    # Training loop
    # First, pretrain public model if using CIFAR-10
    if Config.DATASET_TYPE == 'cifar10':
        logger.info('Pretraining Public Model First')
        
        # Create public model (same architecture as private models)
        public_model = net_list[0]  # Use first model as template
        public_model = nn.DataParallel(public_model, device_ids=device_ids).to(device)
        
        # Setup public data loaders
        public_train_dl, public_train_ds = get_cifar10_dataloader(
            datadir=Config.PRIVATE_DATASET_DIR,
            train_bs=Config.TRAIN_BATCH_SIZE_PRIVATE,
            test_bs=Config.TEST_BATCH_SIZE_PRIVATE,
            dataidxs=public_train_dataidx,
            train_flag=True
        )
        
        public_test_dl, public_test_ds = get_cifar10_dataloader(
            datadir=Config.PRIVATE_DATASET_DIR,
            train_bs=Config.TRAIN_BATCH_SIZE_PRIVATE,
            test_bs=Config.TEST_BATCH_SIZE_PRIVATE,
            dataidxs=public_test_dataidx,
            train_flag=False
        )
        
        logger.info(f'Public model training with {len(public_train_ds)} training samples, {len(public_test_ds)} test samples')
        
        # Train public model
        public_model = pretrain_network(
            epoch=Config.PRETRAIN_EPOCH,
            model=public_model,
            train_dataloader=public_train_dl,
            eval_dataloader=public_test_dl,
            loss_function=Config.PARTICIPANT_PARAMS['loss_function'],
            optimizer_name=Config.PARTICIPANT_PARAMS['optimizer_name'],
            learning_rate=Config.PARTICIPANT_PARAMS['learning_rate'],
            num_classes=Config.OUTPUT_CHANNEL,
            device=device,
            logger=logger,
            metrics_logger=metrics_logger,
            client_id=-1,
        )
        
        # Evaluate and save public model (best checkpoint restored above)
        eval_acc, eval_auc = evaluate_network(public_model, public_test_dl, device, logger)
        logger.info(
            f'Public model (best) evaluation accuracy: {eval_acc:.2f}%, AUC: {eval_auc:.4f}'
        )
        metrics_logger.log(
            phase='public_final_eval', client_id=-1, accuracy=eval_acc, auc=eval_auc
        )
        
        public_save_path = (
            f'{getattr(Config, "MODEL_STORAGE_ROOT", "./Model_Storage")}/{Config.PARTICIPANT_PARAMS["loss_function"]}/'
            f'{Config.NOISE_TYPE}{Config.NOISE_RATE}/public_model.ckpt'
        )
        torch.save(public_model.state_dict(), public_save_path)
        logger.info(f'Saved public model to {public_save_path}')
        
        # Clear public model from GPU
        clear_gpu(public_model)
    
    logger.info(
        f'Pretraining Participants Models '
        f'(parallel_clients={Config.PARALLEL_CLIENTS}, parties={party_ids})'
    )

    def pretrain_one_party(index: int):
        """Train/eval/save one party (server teacher or client)."""
        role = 'server_teacher' if Config.is_server_teacher(index) else 'client'
        ds_key = Config.party_dataset_key(index)

        if Config.DATASET_TYPE in ('genimage', 'diffusionface'):
            root_full = Config.party_datadir(index)
            if not Config.is_server_teacher(index):
                check_genimage_client_val(root_full, logger=logger)
            train_dl_local, train_ds_local = get_dataloader(
                dataset=Config.PRIVATE_DATASET_NAME,
                datadir=root_full,
                train_bs=Config.TRAIN_BATCH_SIZE_PRIVATE,
                test_bs=Config.TEST_BATCH_SIZE_PRIVATE,
                dataidxs=None,
                noise_type=Config.NOISE_TYPE,
                noise_rate=Config.NOISE_RATE,
                train_flag=True
            )
            test_dl, test_ds = get_dataloader(
                dataset=Config.PRIVATE_DATASET_NAME,
                datadir=root_full,
                train_bs=Config.TRAIN_BATCH_SIZE_PRIVATE,
                test_bs=Config.TEST_BATCH_SIZE_PRIVATE,
                dataidxs=None,
                noise_type=Config.NOISE_TYPE,
                noise_rate=Config.NOISE_RATE,
                train_flag=False
            )
        elif Config.DATASET_TYPE == 'cifar10':
            private_dataidx = net_dataidx_map[index]
            train_dl_local, train_ds_local = get_cifar10_dataloader(
                datadir=Config.PRIVATE_DATASET_DIR,
                train_bs=Config.TRAIN_BATCH_SIZE_PRIVATE,
                test_bs=Config.TEST_BATCH_SIZE_PRIVATE,
                dataidxs=private_dataidx,
                train_flag=True
            )
            test_dataidx = test_dataidx_map[index] if test_dataidx_map is not None else None
            test_dl, test_ds = get_cifar10_dataloader(
                datadir=Config.PRIVATE_DATASET_DIR,
                train_bs=Config.TRAIN_BATCH_SIZE_PRIVATE,
                test_bs=Config.TEST_BATCH_SIZE_PRIVATE,
                dataidxs=test_dataidx,
                train_flag=False
            )
        else:
            train_dl_local, train_ds_local = get_dataloader(
                dataset=Config.PRIVATE_DATASET_NAME,
                datadir=Config.PRIVATE_DATASET_DIR,
                train_bs=Config.TRAIN_BATCH_SIZE_PRIVATE,
                test_bs=Config.TEST_BATCH_SIZE_PRIVATE,
                dataidxs=None,
                noise_type=Config.NOISE_TYPE,
                noise_rate=Config.NOISE_RATE,
                train_flag=True
            )
            test_dl, test_ds = get_dataloader(
                dataset=Config.PRIVATE_DATASET_NAME,
                datadir=Config.PRIVATE_DATASET_DIR,
                train_bs=Config.TRAIN_BATCH_SIZE_PRIVATE,
                test_bs=Config.TEST_BATCH_SIZE_PRIVATE,
                dataidxs=None,
                noise_type=Config.NOISE_TYPE,
                noise_rate=Config.NOISE_RATE,
                train_flag=False
            )

        logger.info(
            f'Dataset: {Config.DATASET_TYPE}, party {index} ({role}, {ds_key}), '
            f'Training samples: {len(train_ds_local)}, Test samples: {len(test_ds)}'
        )

        # Avoid DataParallel when multiple clients share one GPU via threads.
        network = net_list[index].to(device)
        netname = Config.PRIVATE_NETS_NAME_LIST[index]
        logger.info(f'Pretraining model {index} ({role}/{ds_key}) with {len(train_ds_local)} samples')
        network = pretrain_network(
            epoch=Config.PRETRAIN_EPOCH,
            model=network,
            train_dataloader=train_dl_local,
            eval_dataloader=test_dl,
            loss_function=Config.PARTICIPANT_PARAMS['loss_function'],
            optimizer_name=Config.PARTICIPANT_PARAMS['optimizer_name'],
            learning_rate=Config.PARTICIPANT_PARAMS['learning_rate'],
            num_classes=Config.OUTPUT_CHANNEL,
            device=device,
            logger=logger,
            metrics_logger=metrics_logger,
            client_id=index,
        )

        eval_acc, eval_auc = evaluate_network(network, test_dl, device, logger)
        logger.info(
            f'Training Evaluation accuracy on model {index} ({role}, best): '
            f'{eval_acc:.2f}%, AUC: {eval_auc:.4f}'
        )
        metrics_logger.log(
            phase='server_teacher_final_eval' if role == 'server_teacher' else 'client_final_eval',
            client_id=index,
            accuracy=eval_acc,
            auc=eval_auc,
        )

        save_path = (
            f'{getattr(Config, "MODEL_STORAGE_ROOT", "./Model_Storage")}/{Config.PARTICIPANT_PARAMS["loss_function"]}/'
            f'{Config.NOISE_TYPE}{Config.NOISE_RATE}/{netname}_{index}.ckpt'
        )
        # Save plain state_dict (no DataParallel wrapper).
        torch.save(network.state_dict(), save_path)
        logger.info(f'Saved model {index} ({role}/{ds_key}) to {save_path}')
        network = network.cpu()
        net_list[index] = network
        torch.cuda.empty_cache()
        return index

    for chunk in iter_chunks(party_ids, Config.PARALLEL_CLIENTS):
        logger.info(f'Pretraining party chunk: {list(chunk)}')
        map_parallel(pretrain_one_party, chunk, max_workers=len(chunk))
        torch.cuda.empty_cache()

    # Final evaluation
    logger.info(f'Evaluating Models (parallel_eval={Config.PARALLEL_EVAL})')
    eval_dl_list = []
    for index in party_ids:
        if Config.DATASET_TYPE in ('genimage', 'diffusionface'):
            root_full = Config.party_datadir(index)
            test_dl, _ = get_dataloader(
                dataset=Config.PRIVATE_DATASET_NAME,
                datadir=root_full,
                train_bs=Config.TRAIN_BATCH_SIZE_PRIVATE,
                test_bs=Config.TEST_BATCH_SIZE_PRIVATE,
                dataidxs=None,
                train_flag=False
            )
        elif Config.DATASET_TYPE == 'cifar10':
            test_dataidx = test_dataidx_map[index] if test_dataidx_map is not None else None
            test_dl, _ = get_cifar10_dataloader(
                datadir=Config.PRIVATE_DATASET_DIR,
                train_bs=Config.TRAIN_BATCH_SIZE_PRIVATE,
                test_bs=Config.TEST_BATCH_SIZE_PRIVATE,
                dataidxs=test_dataidx,
                train_flag=False
            )
        else:
            test_dl, _ = get_dataloader(
                dataset=Config.PRIVATE_DATASET_NAME,
                datadir=Config.PRIVATE_DATASET_DIR,
                train_bs=Config.TRAIN_BATCH_SIZE_PRIVATE,
                test_bs=Config.TEST_BATCH_SIZE_PRIVATE,
                dataidxs=None,
                train_flag=False
            )
        eval_dl_list.append(test_dl)

    def evaluate_one_client(index: int):
        test_accuracy_list = []
        test_auc_list = []
        network = net_list[index]
        netname = Config.PRIVATE_NETS_NAME_LIST[index]
        model_path = (
            f'{getattr(Config, "MODEL_STORAGE_ROOT", "./Model_Storage")}/{Config.PARTICIPANT_PARAMS["loss_function"]}/'
            f'{Config.NOISE_TYPE}{Config.NOISE_RATE}/{netname}_{index}.ckpt'
        )
        state = torch.load(model_path, map_location='cpu')
        # Support both plain and legacy DataParallel checkpoints.
        if any(k.startswith('module.') for k in state.keys()):
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
        else:
            network = network.to(device)
        network.load_state_dict(state)

        for test_dl in eval_dl_list:
            accuracy, auc = evaluate_network(network, test_dl, device, logger)
            test_accuracy_list.append(accuracy)
            test_auc_list.append(auc)

        avg_accuracy = np.mean(test_accuracy_list)
        avg_auc = np.nanmean(test_auc_list)
        logger.info(
            f'Average accuracy for {netname}{index}: {avg_accuracy:.2f}%, '
            f'Average AUC: {avg_auc:.4f}'
        )
        metrics_logger.log(
            phase='cross_eval_avg',
            client_id=index,
            accuracy=avg_accuracy,
            auc=avg_auc,
        )
        network = network.cpu()
        net_list[index] = network.module if isinstance(network, nn.DataParallel) else network
        torch.cuda.empty_cache()
        return index

    for chunk in iter_chunks(party_ids, Config.PARALLEL_EVAL):
        logger.info(f'Eval party chunk: {list(chunk)}')
        map_parallel(evaluate_one_client, chunk, max_workers=len(chunk))
        torch.cuda.empty_cache()

    logger.info(f'Pretrain finished. Text log: {logger.logfile}')
    logger.info(f'Metrics CSV: {metrics_logger.path}')
    metrics_logger.close()

if __name__ == '__main__':
    main()
