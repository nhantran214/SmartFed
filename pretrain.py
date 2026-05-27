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

from Dataset.utils import init_logs, get_dataloader, init_nets, mkdirs
from Dataset.init_dataset import genimage
from Dataset.cifar10_dataset import CIFAR10Dataset, get_cifar10_transforms
from Network.Models_Def.resnet import ResNet10, ResNet12
from loss import SCELoss
from config import Config


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
    device_ids = [0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
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
    logger: Any
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
        
    Returns:
        Module: Trained neural network
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
    
    # Training loop
    model.train()
    for _epoch in range(epoch):
        log_interval = 100
        
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Logging
            if batch_idx % log_interval == 0:
                logger.info(
                    f'Train Epoch: {_epoch} [{batch_idx * len(images)}/{len(train_dataloader.dataset)} '
                    f'({100. * batch_idx / len(train_dataloader):.0f}%)]\tLoss: {loss.item():.6f}'
                )
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in eval_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            logger.info(f'Evaluate Accuracy: {100 * correct / total:.2f}%')
    
    return model

def evaluate_network(
    model: Module,
    dataloader: DataLoader,
    device: torch.device,
    logger: Any
) -> float:
    """Evaluate a neural network.
    
    Args:
        model: Neural network model
        dataloader: Data loader for evaluation
        device: Device to use for evaluation
        logger: Logger for logging evaluation results
        
    Returns:
        float: Accuracy percentage
    """
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        logger.info(f'Test Accuracy: {accuracy:.2f}%')
        return accuracy

def main():
    """Main training function."""
    # Setup logging and device
    mkdirs(f'./Model_Storage/{Config.PARTICIPANT_PARAMS["loss_function"]}/{Config.NOISE_TYPE}{Config.NOISE_RATE}')
    logger = init_logs()
    device, device_ids = setup_device(Config.SEED)
    
    logger.info(f"Loading Participants' Data and Model for {Config.DATASET_TYPE}")
    
    # Initialize networks
    if Config.INTRA_DATASET:
        net_dataidx_map = {}
        for index in range(Config.N_CLIENTS):
            idxes = np.random.permutation(50000)
            idxes = idxes[0:Config.PRIVATE_DATA_LEN]
            net_dataidx_map[index] = idxes
        logger.info(net_dataidx_map)
        net_list = init_nets(
            n_parties=Config.N_CLIENTS,
            nets_name_list=Config.NETS_NAME_LIST,
            num_classes=Config.OUTPUT_CHANNEL
        )
    else:
        if Config.DATASET_TYPE == 'cifar10':
            # For CIFAR-10, create different data splits for each client and public data
            net_dataidx_map = {}
            test_dataidx_map = {}
            public_train_dataidx = None
            public_test_dataidx = None
            
            # Training data distribution (50,000 samples)
            total_train_samples = 50000
            # Reserve some samples for public training (e.g., 10% = 5,000 samples)
            public_train_samples = int(total_train_samples * 0.1)  # 5,000 samples
            local_train_samples = total_train_samples - public_train_samples  # 45,000 samples
            train_samples_per_client = local_train_samples // Config.N_CLIENTS
            
            # Test data distribution (10,000 samples)
            total_test_samples = 10000
            # Reserve some samples for public testing (e.g., 10% = 1,000 samples)
            public_test_samples = int(total_test_samples * 0.1)  # 1,000 samples
            local_test_samples = total_test_samples - public_test_samples  # 9,000 samples
            test_samples_per_client = local_test_samples // Config.N_CLIENTS
            
            # Randomize training and test indices before splitting
            np.random.seed(Config.SEED)  # Ensure reproducibility
            train_indices = np.random.permutation(total_train_samples)
            test_indices = np.random.permutation(total_test_samples)
            
            # Split public data first
            public_train_dataidx = train_indices[:public_train_samples]
            local_train_indices = train_indices[public_train_samples:]
            
            public_test_dataidx = test_indices[:public_test_samples]
            local_test_indices = test_indices[public_test_samples:]
            
            for index in range(Config.N_CLIENTS):
                # Training data split (randomized, excluding public data)
                train_start_idx = index * train_samples_per_client
                train_end_idx = train_start_idx + train_samples_per_client
                if index == Config.N_CLIENTS - 1:  # Last client gets remaining samples
                    train_end_idx = local_train_samples
                train_idxes = local_train_indices[train_start_idx:train_end_idx]
                net_dataidx_map[index] = train_idxes
                
                # Test data split (randomized, excluding public data)
                test_start_idx = index * test_samples_per_client
                test_end_idx = test_start_idx + test_samples_per_client
                if index == Config.N_CLIENTS - 1:  # Last client gets remaining samples
                    test_end_idx = local_test_samples
                test_idxes = local_test_indices[test_start_idx:test_end_idx]
                test_dataidx_map[index] = test_idxes
            
            logger.info(f"CIFAR-10 data split:")
            logger.info(f"  - Public train: {len(public_train_dataidx)} samples")
            logger.info(f"  - Public test: {len(public_test_dataidx)} samples")
            logger.info(f"  - Local train: {[(k, len(v)) for k, v in net_dataidx_map.items()]}")
            logger.info(f"  - Local test: {[(k, len(v)) for k, v in test_dataidx_map.items()]}")
            logger.info(f"Random seed used: {Config.SEED}")
        else:
            net_dataidx_map = {index: None for index in range(Config.N_CLIENTS)}
            logger.info(net_dataidx_map)
        
        net_list = init_nets(
            n_parties=Config.N_CLIENTS,
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
            epoch=20,  # Same as private models
            model=public_model,
            train_dataloader=public_train_dl,
            eval_dataloader=public_test_dl,
            loss_function=Config.PARTICIPANT_PARAMS['loss_function'],
            optimizer_name=Config.PARTICIPANT_PARAMS['optimizer_name'],
            learning_rate=Config.PARTICIPANT_PARAMS['learning_rate'],
            num_classes=Config.OUTPUT_CHANNEL,
            device=device,
            logger=logger
        )
        
        # Evaluate and save public model
        eval_acc = evaluate_network(public_model, public_test_dl, device, logger)
        logger.info(f'Public model evaluation accuracy: {eval_acc:.2f}%')
        
        public_save_path = (
            f'./Model_Storage/{Config.PARTICIPANT_PARAMS["loss_function"]}/'
            f'{Config.NOISE_TYPE}{Config.NOISE_RATE}/public_model.ckpt'
        )
        torch.save(public_model.state_dict(), public_save_path)
        logger.info(f'Saved public model to {public_save_path}')
        
        # Clear public model from GPU
        clear_gpu(public_model)
    
    logger.info('Pretraining Participants Models')
    for index in range(Config.N_CLIENTS):
        # Setup data loaders based on dataset type
        if Config.DATASET_TYPE == 'genimage':
            root_full = os.path.join(Config.PRIVATE_DATASET_DIR, Config.DATASET_LIST[index])
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
            # For CIFAR-10, use data selection for different clients
            private_dataidx = net_dataidx_map[index]
            train_dl_local, train_ds_local = get_cifar10_dataloader(
                datadir=Config.PRIVATE_DATASET_DIR,
                train_bs=Config.TRAIN_BATCH_SIZE_PRIVATE,
                test_bs=Config.TEST_BATCH_SIZE_PRIVATE,
                dataidxs=private_dataidx,
                train_flag=True
            )
            
            # Test dataset uses distributed test data for each client
            test_dataidx = test_dataidx_map[index] if test_dataidx_map is not None else None
            test_dl, test_ds = get_cifar10_dataloader(
                datadir=Config.PRIVATE_DATASET_DIR,
                train_bs=Config.TRAIN_BATCH_SIZE_PRIVATE,
                test_bs=Config.TEST_BATCH_SIZE_PRIVATE,
                dataidxs=test_dataidx,
                train_flag=False
            )
        else:
            # Fallback to original method
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
        
        logger.info(f'Dataset: {Config.DATASET_TYPE}, Client {index}, '
                   f'Training samples: {len(train_ds_local)}, Test samples: {len(test_ds)}')
        
        # Train model
        network = net_list[index]
        network = nn.DataParallel(network, device_ids=device_ids).to(device)
        netname = Config.PRIVATE_NETS_NAME_LIST[index]
        
        logger.info(f'Pretraining model {index} with {len(train_ds_local)} training samples')
        network = pretrain_network(
            epoch=20,  # Fixed pretrain epochs
            model=network,
            train_dataloader=train_dl_local,
            eval_dataloader=test_dl,
            loss_function=Config.PARTICIPANT_PARAMS['loss_function'],
            optimizer_name=Config.PARTICIPANT_PARAMS['optimizer_name'],
            learning_rate=Config.PARTICIPANT_PARAMS['learning_rate'],
            num_classes=Config.OUTPUT_CHANNEL,
            device=device,
            logger=logger
        )
        
        # Evaluate and save model
        eval_acc = evaluate_network(network, test_dl, device, logger)
        logger.info(f'Training Evaluation accuracy on model {index}: {eval_acc:.2f}%')
        
        save_path = (
            f'./Model_Storage/{Config.PARTICIPANT_PARAMS["loss_function"]}/'
            f'{Config.NOISE_TYPE}{Config.NOISE_RATE}/{netname}_{index}.ckpt'
        )
        torch.save(network.state_dict(), save_path)
        logger.info(f'Saved model {index} to {save_path}')
    
    # Final evaluation
    logger.info('Evaluating Models')
    eval_dl_list = []
    for index in range(Config.N_CLIENTS):
        if Config.DATASET_TYPE == 'genimage':
            root_full = os.path.join(Config.PRIVATE_DATASET_DIR, Config.DATASET_LIST[index])
            test_dl, _ = get_dataloader(
                dataset=Config.PRIVATE_DATASET_NAME,
                datadir=root_full,
                train_bs=Config.TRAIN_BATCH_SIZE_PRIVATE,
                test_bs=Config.TEST_BATCH_SIZE_PRIVATE,
                dataidxs=None,
                train_flag=False
            )
        elif Config.DATASET_TYPE == 'cifar10':
            # Use distributed test data for each client
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
    
    for index in range(Config.N_CLIENTS):
        test_accuracy_list = []
        network = net_list[index]
        network = nn.DataParallel(network, device_ids=device_ids).to(device)
        netname = Config.PRIVATE_NETS_NAME_LIST[index]
        
        model_path = (
            f'./Model_Storage/{Config.PARTICIPANT_PARAMS["loss_function"]}/'
            f'{Config.NOISE_TYPE}{Config.NOISE_RATE}/{netname}_{index}.ckpt'
        )
        network.load_state_dict(torch.load(model_path))
        
        for test_dl in eval_dl_list:
            accuracy = evaluate_network(network, test_dl, device, logger)
            test_accuracy_list.append(accuracy)
        
        avg_accuracy = np.mean(test_accuracy_list)
        logger.info(f'Average accuracy for {netname}{index}: {avg_accuracy:.2f}%')

if __name__ == '__main__':
    main()
