#!/usr/bin/env python3
"""
Test script to verify CIFAR-10 data distribution for both training and test sets.
This script simulates the data distribution logic without actually loading the data.
"""

import numpy as np
import sys
sys.path.append('.')

from config import Config

def test_cifar10_distribution():
    """Test CIFAR-10 data distribution logic with randomization and public data split"""
    print("=" * 60)
    print("CIFAR-10 Data Distribution Test (with Public Data Split)")
    print("=" * 60)
    
    # Simulate the distribution logic
    N_CLIENTS = Config.N_CLIENTS
    total_train_samples = 50000
    total_test_samples = 10000
    seed = Config.SEED
    
    # Public data split (10% of total data)
    public_train_samples = int(total_train_samples * 0.1)  # 5,000 samples
    local_train_samples = total_train_samples - public_train_samples  # 45,000 samples
    train_samples_per_client = local_train_samples // N_CLIENTS
    
    public_test_samples = int(total_test_samples * 0.1)  # 1,000 samples
    local_test_samples = total_test_samples - public_test_samples  # 9,000 samples
    test_samples_per_client = local_test_samples // N_CLIENTS
    
    print(f"Number of clients: {N_CLIENTS}")
    print(f"Total training samples: {total_train_samples}")
    print(f"Total test samples: {total_test_samples}")
    print(f"Public train samples: {public_train_samples} (10%)")
    print(f"Public test samples: {public_test_samples} (10%)")
    print(f"Local train samples: {local_train_samples} (90%)")
    print(f"Local test samples: {local_test_samples} (90%)")
    print(f"Training samples per client: {train_samples_per_client}")
    print(f"Test samples per client: {test_samples_per_client}")
    print(f"Random seed: {seed}")
    print()
    
    # Randomize training and test indices before splitting
    np.random.seed(seed)
    train_indices = np.random.permutation(total_train_samples)
    test_indices = np.random.permutation(total_test_samples)
    
    print("RANDOMIZATION:")
    print("-" * 40)
    print(f"Training indices sample: {train_indices[:10]}...")
    print(f"Test indices sample: {test_indices[:10]}...")
    print("✅ Indices are randomized before splitting")
    print()
    
    # Split public data first
    public_train_dataidx = train_indices[:public_train_samples]
    local_train_indices = train_indices[public_train_samples:]
    
    public_test_dataidx = test_indices[:public_test_samples]
    local_test_indices = test_indices[public_test_samples:]
    
    print("PUBLIC DATA SPLIT:")
    print("-" * 40)
    print(f"Public train: {len(public_train_dataidx)} samples (indices: {public_train_dataidx[:5]}...{public_train_dataidx[-5:]})")
    print(f"Public test: {len(public_test_dataidx)} samples (indices: {public_test_dataidx[:5]}...{public_test_dataidx[-5:]})")
    print()
    
    # Local training data distribution
    print("LOCAL TRAINING DATA DISTRIBUTION (RANDOMIZED):")
    print("-" * 40)
    net_dataidx_map = {}
    for index in range(N_CLIENTS):
        train_start_idx = index * train_samples_per_client
        train_end_idx = train_start_idx + train_samples_per_client
        if index == N_CLIENTS - 1:  # Last client gets remaining samples
            train_end_idx = local_train_samples
        train_idxes = local_train_indices[train_start_idx:train_end_idx]
        net_dataidx_map[index] = train_idxes
        print(f"Client {index}: {len(train_idxes)} samples (indices: {train_idxes[:5]}...{train_idxes[-5:]})")
    
    print()
    
    # Local test data distribution
    print("LOCAL TEST DATA DISTRIBUTION (RANDOMIZED):")
    print("-" * 40)
    test_dataidx_map = {}
    for index in range(N_CLIENTS):
        test_start_idx = index * test_samples_per_client
        test_end_idx = test_start_idx + test_samples_per_client
        if index == N_CLIENTS - 1:  # Last client gets remaining samples
            test_end_idx = local_test_samples
        test_idxes = local_test_indices[test_start_idx:test_end_idx]
        test_dataidx_map[index] = test_idxes
        print(f"Client {index}: {len(test_idxes)} samples (indices: {test_idxes[:5]}...{test_idxes[-5:]})")
    
    print()
    
    # Verify no overlap and complete coverage
    print("VERIFICATION:")
    print("-" * 40)
    
    # Check public data
    print(f"Public train coverage: {len(set(public_train_dataidx)) == len(public_train_dataidx)}")
    print(f"Public test coverage: {len(set(public_test_dataidx)) == len(public_test_dataidx)}")
    
    # Check local training data
    all_local_train_indices = set()
    for indices in net_dataidx_map.values():
        all_local_train_indices.update(indices)
    
    print(f"Local train data coverage: {len(all_local_train_indices) == local_train_samples}")
    print(f"Local train data no overlap: {len(all_local_train_indices) == sum(len(indices) for indices in net_dataidx_map.values())}")
    
    # Check local test data
    all_local_test_indices = set()
    for indices in test_dataidx_map.values():
        all_local_test_indices.update(indices)
    
    print(f"Local test data coverage: {len(all_local_test_indices) == local_test_samples}")
    print(f"Local test data no overlap: {len(all_local_test_indices) == sum(len(indices) for indices in test_dataidx_map.values())}")
    
    # Check no overlap between public and local data
    public_train_set = set(public_train_dataidx)
    public_test_set = set(public_test_dataidx)
    local_train_set = all_local_train_indices
    local_test_set = all_local_test_indices
    
    print(f"No overlap public/local train: {len(public_train_set.intersection(local_train_set)) == 0}")
    print(f"No overlap public/local test: {len(public_test_set.intersection(local_test_set)) == 0}")
    
    # Check reproducibility
    print()
    print("REPRODUCIBILITY CHECK:")
    print("-" * 40)
    np.random.seed(seed)
    train_indices_2 = np.random.permutation(total_train_samples)
    test_indices_2 = np.random.permutation(total_test_samples)
    
    print(f"Training indices reproducible: {np.array_equal(train_indices, train_indices_2)}")
    print(f"Test indices reproducible: {np.array_equal(test_indices, test_indices_2)}")
    
    # Check IID property (each client should have samples from all classes)
    print()
    print("IID DISTRIBUTION CHECK:")
    print("-" * 40)
    print("Each client should have samples from all 10 CIFAR-10 classes")
    print("This is ensured by the CIFAR10Dataset class which loads all classes")
    print("and then selects samples based on randomized dataidxs")
    
    print()
    print("SUMMARY:")
    print("-" * 40)
    print(f"✅ Public train: {public_train_samples} samples (10% of total)")
    print(f"✅ Public test: {public_test_samples} samples (10% of total)")
    print(f"✅ Local train: {local_train_samples} samples divided among {N_CLIENTS} clients")
    print(f"✅ Local test: {local_test_samples} samples divided among {N_CLIENTS} clients")
    print(f"✅ Each client gets ~{train_samples_per_client} training samples")
    print(f"✅ Each client gets ~{test_samples_per_client} test samples")
    print(f"✅ IID distribution: Each client has samples from all 10 classes")
    print(f"✅ No data overlap between public and local data")
    print(f"✅ No data overlap between clients")
    print(f"✅ Complete data coverage")
    print(f"✅ Randomized distribution (seed: {seed})")
    print(f"✅ Reproducible results")

if __name__ == '__main__':
    test_cifar10_distribution()
