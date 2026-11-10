import sys
sys.path.append("..")
sys.path.append('./')
# Increasing recursion limit
sys.setrecursionlimit(10000)

# Standard library imports
import os
import random
import numpy as np
from numpy import mean
import copy

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn
from tensorboardX import SummaryWriter

# Local imports
from Dataset.utils import (
    init_logs, get_dataloader, init_nets, mkdirs, compute_accuracy_and_auc,
    MetricsLogger, log_run_config, across_is_better, save_best_checkpoints,
)
from loss import SCELoss
from config import Config
from parallel_utils import iter_chunks, map_parallel

def clear_gpu(model):
    """Clear GPU memory"""
    model = model.to('cpu')
    del model
    torch.cuda.empty_cache()

def clear_image(images, labels):
    """Clear image and label tensors from GPU"""
    del images, labels
    torch.cuda.empty_cache()

def clear_criterion(criterion):
    """Clear criterion from GPU"""
    del criterion
    torch.cuda.empty_cache()

def evaluate_model(model, dataloader, logger, device):
    """Evaluate model performance on a dataloader (Accuracy + AUC)."""
    acc, auc = compute_accuracy_and_auc(model, dataloader, device)
    logger.info(f'Test Accuracy: {acc:.2f}%, AUC: {auc:.4f}')
    return acc, auc

def local_training(model, train_dataloader, loss_function, optimizer_method, 
                  learning_rate, local_epochs, logger, device):
    """Perform local training on a client's data"""
    # Initialize loss function
    if loss_function == 'CE':
        criterion = nn.CrossEntropyLoss()
    elif loss_function == 'SCE':
        criterion = SCELoss(alpha=0.1, beta=1.0, num_classes=2)
    
    # Initialize optimizer
    if optimizer_method == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_method == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    
    model.train()
    local_losses = []
    local_accuracies = []
    
    for epoch in range(local_epochs):
        epoch_loss = 0.0
        num_batches = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                batch_acc = 100 * correct / total
                logger.info(f'Local Epoch {epoch+1}/{local_epochs}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.6f}, Acc: {batch_acc:.2f}%')
        
        avg_epoch_loss = epoch_loss / num_batches
        epoch_accuracy = 100 * correct / total
        local_losses.append(avg_epoch_loss)
        local_accuracies.append(epoch_accuracy)
        logger.info(f'Local Epoch {epoch+1}/{local_epochs} completed, Average Loss: {avg_epoch_loss:.6f}, Accuracy: {epoch_accuracy:.2f}%')
    
    return model, local_losses, local_accuracies

def fedavg_aggregate(models, weights=None):
    """Aggregate models using FedAvg algorithm"""
    if weights is None:
        # Equal weights for all models
        weights = [1.0 / len(models)] * len(models)
    
    # Get the state dict of the first model as template
    # Handle DataParallel models by accessing .module
    first_model = models[0]
    if isinstance(first_model, nn.DataParallel):
        aggregated_state = first_model.module.state_dict()
    else:
        aggregated_state = first_model.state_dict()
    
    # Initialize aggregated parameters
    for key in aggregated_state.keys():
        aggregated_state[key] = torch.zeros_like(aggregated_state[key])
    
    # Weighted average of all model parameters
    for model, weight in zip(models, weights):
        if isinstance(model, nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
        
        for key in aggregated_state.keys():
            # Only aggregate parameters that are trainable (float tensors)
            if aggregated_state[key].dtype.is_floating_point:
                aggregated_state[key] += weight * model_state[key]
            else:
                # For non-float parameters (like batch norm running stats), use the first model's values
                if model == models[0]:  # Only set once
                    aggregated_state[key] = model_state[key].clone()
    
    # Create new model with aggregated parameters
    # Use deep copy of the first model to preserve structure
    if isinstance(first_model, nn.DataParallel):
        # Get the original model from DataParallel
        original_model = first_model.module
        aggregated_model = copy.deepcopy(original_model)
    else:
        aggregated_model = copy.deepcopy(first_model)
    
    # Load the aggregated state dict
    aggregated_model.load_state_dict(aggregated_state)
    
    return aggregated_model

def across_evaluation(models, N_CLIENTS, Private_Dataset_Name, Private_Dataset_Dir, 
                     TrainBatchSize, TestBatchSize, Noise_type, Noise_rate, logger, device,
                     metrics_logger=None, round_idx=None):
    """Evaluate models across all participants"""
    test_list = []
    logger.info('Across Evaluate Models')

    for c_index in range(N_CLIENTS):
        if Private_Dataset_Name in ('genimage', 'diffusionface'):
            root_genim_full = os.path.join(Private_Dataset_Dir, Config.DATASET_LIST[c_index])
        test_dl, _ = get_dataloader(dataset=Private_Dataset_Name, 
                                   datadir=root_genim_full,
                                   train_bs=Config.TRAIN_BATCH_SIZE_PRIVATE,
                                   test_bs=Config.TEST_BATCH_SIZE_PRIVATE,
                                   dataidxs=None,
                                   noise_type=Noise_type,
                                   noise_rate=Noise_rate)
        test_list.append(test_dl)

    def evaluate_one_model(m):
        model = models[m].to(device)
        acc_model_list = []
        auc_model_list = []
        for i in range(N_CLIENTS):
            accuracy, auc = evaluate_model(
                model=model, dataloader=test_list[i], logger=logger, device=device
            )
            acc_model_list.append(accuracy)
            auc_model_list.append(auc)
            logger.info(
                f'Test Accuracy of model {m} on dataset {i}: {accuracy:.2f}%, '
                f'AUC: {auc:.4f}'
            )
        accuracy_avg = sum(acc_model_list) / N_CLIENTS
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
        models[m] = model.cpu()
        torch.cuda.empty_cache()
        return m, accuracy_avg, auc_avg

    scores = []
    model_ids = list(range(len(models)))
    for chunk in iter_chunks(model_ids, getattr(Config, 'PARALLEL_EVAL', 1)):
        logger.info(f'Across-eval model chunk: {list(chunk)}')
        scores.extend(map_parallel(evaluate_one_model, chunk, max_workers=len(chunk)))
        torch.cuda.empty_cache()
    if not scores:
        return None, None
    mean_acc = sum(acc for _, acc, _ in scores) / len(scores)
    mean_auc = float(np.nanmean([auc for _, _, auc in scores]))
    return mean_acc, mean_auc

def save_models_and_metrics(models, config, logger, epoch_index):
    """Save models and training metrics"""
    base_path = f'{getattr(config, "MODEL_STORAGE_ROOT", "./Model_Storage")}/{config.PARTICIPANT_PARAMS["loss_function"]}/{config.NOISE_TYPE}{config.NOISE_RATE}/'
    mkdirs(base_path)
    
    for i, model in enumerate(models):
        netname = config.PRIVATE_NETS_NAME_LIST[i]
        # Handle DataParallel models
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), f'{base_path}{netname}_{i}.ckpt')
        else:
            torch.save(model.state_dict(), f'{base_path}{netname}_{i}.ckpt')
    
    logger.info("Models and metrics saved successfully")

def main():
    """Main FedAvg training function"""
    # Initialize logging
    logger = init_logs()
    metrics_logger = MetricsLogger(run_id=getattr(logger, 'run_id', None))
    logger.info("Initializing FedAvg training configuration")
    logger.info(f'Metrics CSV: {metrics_logger.path}')
    log_run_config(logger, Config)
    
    # Set random seeds
    random.seed(Config.SEED)
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)
    
    # Setup device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device_ids = [0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Initialize models and data
    logger.info("Initializing models and data for FedAvg")
    if Config.INTRA_DATASET:
        net_dataidx_map = {}
        for index in range(Config.N_CLIENTS):
            idxes = np.random.permutation(50000)
            idxes = idxes[0:Config.PRIVATE_DATA_LEN]
            net_dataidx_map[index] = idxes
        logger.info(net_dataidx_map)
        net_list = init_nets(n_parties=Config.N_CLIENTS,
                           nets_name_list=Config.NETS_NAME_LIST,
                           num_classes=Config.OUTPUT_CHANNEL)
    else:
        net_dataidx_map = {index: None for index in range(Config.N_CLIENTS)}
        logger.info(net_dataidx_map)
        net_list = init_nets(n_parties=Config.N_CLIENTS,
                           nets_name_list=Config.NETS_NAME_LIST,
                           num_classes=Config.OUTPUT_CHANNEL)
    
    # Initialize global model
    global_model = net_list[0]
    global_model = nn.DataParallel(global_model, device_ids=device_ids).to(device)
    
    # FedAvg training loop
    logger.info("Starting FedAvg training")
    global_losses = []
    local_losses_all_clients = []
    best_across = {'acc': -1.0, 'auc': -1.0, 'round': None, 'full': False}
    best_dir = (
        f'{getattr(Config, "MODEL_STORAGE_ROOT", "./Model_Storage")}/'
        f'{Config.PARTICIPANT_PARAMS["loss_function"]}/'
        f'{Config.NOISE_TYPE}{Config.NOISE_RATE}/'
    )
    
    for round_idx in range(Config.COMMUNICATION_EPOCH):
        logger.info(f"FedAvg Round {round_idx + 1}/{Config.COMMUNICATION_EPOCH}")
        
        # Local training on each client (parallel chunks)
        local_models = [None] * Config.N_CLIENTS
        round_local_losses = [None] * Config.N_CLIENTS
        round_local_accuracies = [None] * Config.N_CLIENTS
        global_state = global_model.module.state_dict()

        def train_one_client(client_idx):
            logger.info(f"Training client {client_idx + 1}/{Config.N_CLIENTS}")
            local_model = net_list[client_idx]
            local_model.load_state_dict(global_state)
            local_model = local_model.to(device)

            private_dataidx = net_dataidx_map[client_idx]
            if Config.PRIVATE_DATASET_NAME in ('genimage', 'diffusionface'):
                root_genim_full = os.path.join(
                    Config.PRIVATE_DATASET_DIR, Config.DATASET_LIST[client_idx]
                )
            else:
                root_genim_full = Config.PRIVATE_DATASET_DIR

            train_dl_local, train_ds_local = get_dataloader(
                dataset=Config.PRIVATE_DATASET_NAME,
                datadir=root_genim_full,
                train_bs=Config.TRAIN_BATCH_SIZE_PRIVATE,
                test_bs=Config.TEST_BATCH_SIZE_PRIVATE,
                dataidxs=private_dataidx,
                noise_type=Config.NOISE_TYPE,
                noise_rate=Config.NOISE_RATE,
                train_flag=True,
            )
            logger.info(f'Client {client_idx}: Training samples: {len(train_ds_local)}')

            local_model, client_losses, client_accuracies = local_training(
                model=local_model,
                train_dataloader=train_dl_local,
                loss_function=Config.PARTICIPANT_PARAMS['loss_function'],
                optimizer_method=Config.PARTICIPANT_PARAMS['optimizer_name'],
                learning_rate=Config.PARTICIPANT_PARAMS['learning_rate'],
                local_epochs=1,
                logger=logger,
                device=device,
            )
            final_accuracy = client_accuracies[-1] if client_accuracies else 0
            logger.info(f'Client {client_idx} final accuracy: {final_accuracy:.2f}%')
            local_model_cpu = local_model.cpu()
            torch.cuda.empty_cache()
            return client_idx, local_model_cpu, mean(client_losses), final_accuracy

        for chunk in iter_chunks(list(range(Config.N_CLIENTS)), Config.PARALLEL_CLIENTS):
            logger.info(f'FedAvg local-train chunk: {list(chunk)}')
            results = map_parallel(train_one_client, chunk, max_workers=len(chunk))
            for client_idx, model_cpu, loss_val, acc_val in results:
                local_models[client_idx] = model_cpu
                round_local_losses[client_idx] = loss_val
                round_local_accuracies[client_idx] = acc_val
            torch.cuda.empty_cache()
        
        # FedAvg aggregation
        logger.info("Aggregating models using FedAvg")
        aggregated_model = fedavg_aggregate(local_models)
        global_model = nn.DataParallel(aggregated_model, device_ids=device_ids).to(device)
        
        # Store losses and accuracies
        local_losses_all_clients.append(round_local_losses)
        global_losses.append(mean(round_local_losses))
        
        avg_accuracy = mean(round_local_accuracies)
        logger.info(f"Round {round_idx + 1} completed. Average local loss: {mean(round_local_losses):.6f}, Average accuracy: {avg_accuracy:.2f}%")
        
        # Evaluation
        if (round_idx + 1) % 1 == 0 or round_idx == Config.COMMUNICATION_EPOCH - 1:
            logger.info(f"Evaluating global model at round {round_idx + 1}")
            # One global model evaluated across all client test sets (avoid N× duplicate work).
            mean_acc, mean_auc = across_evaluation(
                [global_model], Config.N_CLIENTS, Config.PRIVATE_DATASET_NAME,
                Config.PRIVATE_DATASET_DIR, Config.TRAIN_BATCH_SIZE, Config.TEST_BATCH_SIZE,
                Config.NOISE_TYPE, Config.NOISE_RATE, logger, device,
                metrics_logger=metrics_logger, round_idx=round_idx,
            )
            if mean_acc is not None and across_is_better(mean_acc, mean_auc, True, best_across):
                best_across = {
                    'acc': mean_acc,
                    'auc': mean_auc if mean_auc == mean_auc else -1.0,
                    'round': round_idx,
                    'full': True,
                }
                save_best_checkpoints(
                    [global_model], Config.NETS_NAME_LIST, best_dir,
                    round_idx, mean_acc, mean_auc, True, logger,
                )
            metrics_logger.log(
                phase='round_summary',
                round=round_idx,
                loss=mean(round_local_losses),
                accuracy=avg_accuracy,
            )
        
        # Save models periodically
        if (round_idx + 1) % 10 == 0 or round_idx == Config.COMMUNICATION_EPOCH - 1:
            save_models_and_metrics([global_model], Config, logger, round_idx)
    
    # Final evaluation
    logger.info('Final evaluation of FedAvg model')
    mean_acc, mean_auc = across_evaluation(
        [global_model], Config.N_CLIENTS, Config.PRIVATE_DATASET_NAME,
        Config.PRIVATE_DATASET_DIR, Config.TRAIN_BATCH_SIZE, Config.TEST_BATCH_SIZE,
        Config.NOISE_TYPE, Config.NOISE_RATE, logger, device,
        metrics_logger=metrics_logger, round_idx=Config.COMMUNICATION_EPOCH - 1,
    )
    final_round = Config.COMMUNICATION_EPOCH - 1
    if mean_acc is not None and across_is_better(mean_acc, mean_auc, True, best_across):
        best_across = {
            'acc': mean_acc,
            'auc': mean_auc if mean_auc == mean_auc else -1.0,
            'round': final_round,
            'full': True,
        }
        save_best_checkpoints(
            [global_model], Config.NETS_NAME_LIST, best_dir,
            final_round, mean_acc, mean_auc, True, logger,
        )
    if best_across.get('round') is not None:
        logger.info(
            f'Best across-eval kept: round {best_across["round"]} '
            f'Acc={best_across["acc"]:.2f}%, AUC={best_across["auc"]:.4f}'
        )
    
    # Save final model
    save_models_and_metrics([global_model], Config, logger, Config.COMMUNICATION_EPOCH - 1)
    
    # Print training summary
    logger.info("="*50)
    logger.info("FedAvg Training Summary:")
    logger.info(f"Total rounds: {Config.COMMUNICATION_EPOCH}")
    logger.info(f"Average local loss: {mean(global_losses):.6f}")
    logger.info("="*50)
    logger.info("FedAvg training completed successfully!")
    logger.info(f'Text log: {logger.logfile}')
    logger.info(f'Metrics CSV: {metrics_logger.path}')
    metrics_logger.close()

if __name__ == '__main__':
    main()
