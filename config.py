"""Configuration parameters for SmartFed (federated learning).

Default vision backbone per client: EfficientNet-B0 (`PRIVATE_NETS_NAME_LIST` / `NETS_NAME_LIST`).
"""

import os

# Repository root (directory containing this file). Dataset paths default under `./data/`.
_SMARTFED_ROOT = os.path.dirname(os.path.abspath(__file__))


class Config:
    # Training Parameters
    SEED = 0
    N_CLIENTS = 8
    TRAIN_BATCH_SIZE = 4
    TEST_BATCH_SIZE = 4
    TRAIN_BATCH_SIZE_PRIVATE = 64
    TEST_BATCH_SIZE_PRIVATE = 32
    COMMUNICATION_EPOCH = 40
    INTRA_DATASET = False
    
    # Model Parameters
    PARTICIPANT_PARAMS = {
        'loss_function': 'CE',
        'optimizer_name': 'Adam',
        'learning_rate': 0.001
    }
    
    # Client Confidence Reweight Parameters
    CLIENT_CONFIDENCE_REWEIGHT = True
    CLIENT_CONFIDENCE_REWEIGHT_LOSS = 'CE'
    BETA = 0.5 if CLIENT_CONFIDENCE_REWEIGHT else 0
    
    # Noise Parameters
    NOISE_TYPE = None  # ['pairflip', 'symmetric', None]
    NOISE_RATE = 0
    
    # Model Architecture
    PRIVATE_NETS_NAME_LIST = ['EfficientNetB0'] * N_CLIENTS
    NETS_NAME_LIST = ['EfficientNetB0'] * N_CLIENTS
    
    # Dataset Configuration
    # Choose dataset: 'genimage' or 'cifar10'
    DATASET_TYPE = 'cifar10'  # Change this to 'cifar10' for CIFAR-10 training
    
    if DATASET_TYPE == 'genimage':
        DATASET_LIST = ['adm', 'BigGAN', 'glide', 'midjourney', 
                    'sdv4', 'sdv5', 'vqdm', 'wukong']
        PRIVATE_DATASET_NAME = 'genimage'
        PRIVATE_DATASET_DIR = os.path.join(_SMARTFED_ROOT, 'data', 'genimage_medium_fulltest')
        PRIVATE_DATA_LEN = 5000
        PRIVATE_DATASET_CLASSES = ['real', 'fake']
        OUTPUT_CHANNEL = len(PRIVATE_DATASET_CLASSES)
        PRIVATE_OUTPUT_CHANNEL = len(PRIVATE_DATASET_CLASSES)
        
        # Public Dataset Configuration
        PUBLIC_DATASET_NAME = 'genimage'
        PUBLIC_DATASET_DIR = os.path.join(
            _SMARTFED_ROOT, 'data', 'genimage_medium_fulltest', 'genimage_pub'
        )
        PUBLIC_DATASET_LENGTH = 8000
        
    elif DATASET_TYPE == 'cifar10':
        DATASET_LIST = ['cifar10'] * N_CLIENTS  # All clients use CIFAR-10
        PRIVATE_DATASET_NAME = 'cifar10'
        PRIVATE_DATASET_DIR = os.path.join(_SMARTFED_ROOT, 'data', 'cifar')
        PRIVATE_DATA_LEN = 5000  # Number of samples per client
        PRIVATE_DATASET_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                                  'dog', 'frog', 'horse', 'ship', 'truck']
        OUTPUT_CHANNEL = len(PRIVATE_DATASET_CLASSES)
        PRIVATE_OUTPUT_CHANNEL = len(PRIVATE_DATASET_CLASSES)
        
        # Public Dataset Configuration (same as private for CIFAR-10)
        PUBLIC_DATASET_NAME = 'cifar10'
        PUBLIC_DATASET_DIR = os.path.join(_SMARTFED_ROOT, 'data', 'cifar')
        PUBLIC_DATASET_LENGTH = 8000 