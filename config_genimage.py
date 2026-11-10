"""GenImage / DiffusionFace Config for SmartFed.

Loaded by default via `from config import Config`.
CIFAR-10 lives in `config_cifar.py` (SMARTFED_CONFIG=cifar).
"""

import os

from config_base import (
    ConfigPartyMixin,
    _SMARTFED_ROOT,
    _env_int,
    _env_optional_int,
    _env_str,
    resolve_agent_modes,
)


class Config(ConfigPartyMixin):
    PROTOCOL = _env_str('SMARTFED_PROTOCOL', 'Homo_ServerGSAM')
    SEED = _env_int('SMARTFED_SEED', 0)
    N_CLIENTS = _env_int('SMARTFED_N_CLIENTS', 8)
    TRAIN_BATCH_SIZE = 32
    TEST_BATCH_SIZE = 32
    TRAIN_BATCH_SIZE_PRIVATE = 32
    TEST_BATCH_SIZE_PRIVATE = 32
    COMMUNICATION_EPOCH = _env_int('SMARTFED_COMMUNICATION_EPOCH', 40)
    PRETRAIN_EPOCH = _env_int('SMARTFED_PRETRAIN_EPOCH', 5)
    INTRA_DATASET = False
    PRETRAIN_PARTY_IDS = [8]

    AGENT_MODE = _env_str('SMARTFED_AGENT_MODE', 'llm').lower()
    DCSA_MODE = _env_str('SMARTFED_DCSA_MODE', '').lower()
    AFWA_MODE = _env_str('SMARTFED_AFWA_MODE', '').lower()
    DCSA_RULE_STRATEGY = _env_str('SMARTFED_DCSA_RULE_STRATEGY', 'adaptive').lower()
    SELECTION_BUDGET = _env_optional_int('SMARTFED_SELECTION_BUDGET')
    AFWA_FIXED_COEFFICIENT = float(_env_str('SMARTFED_AFWA_FIXED_COEFFICIENT', '0.1'))
    AFWA_COEFF_MIN = float(_env_str('SMARTFED_AFWA_COEFF_MIN', '0.01'))
    AFWA_COEFF_MAX = float(_env_str('SMARTFED_AFWA_COEFF_MAX', '1.0'))
    LOG_LLM_COST = _env_str('SMARTFED_LOG_LLM_COST', '1').lower() not in ('0', 'false', 'no')
    OLLAMA_MODEL_NAME = _env_str('SMARTFED_OLLAMA_MODEL', 'llama3:8b')

    PARALLEL_CLIENTS = 1
    PARALLEL_EVAL = 1
    REQUIRE_CUDA = True
    KEEP_MODELS_ON_GPU = True
    CUDNN_BENCHMARK = True

    OLLAMA_ON_GPU = True
    OLLAMA_UNLOAD_AFTER_QUERY = True
    OLLAMA_TIMEOUT = _env_int('SMARTFED_OLLAMA_TIMEOUT', 30)
    OLLAMA_MAX_RETRIES = _env_int('SMARTFED_OLLAMA_MAX_RETRIES', 3)
    OLLAMA_MAX_PARALLEL = 1
    OLLAMA_MAX_TOKENS = _env_int('SMARTFED_OLLAMA_MAX_TOKENS', 1000)

    DATALOADER_NUM_WORKERS = 8
    DATA_CACHE_ENABLED = True
    DATA_CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'smartfed', 'images')
    DATA_CACHE_IMAGE_SIZE = 224
    DATA_CACHE_PREWARM = False

    LOCAL_WEIGHT_MAX_BATCHES = _env_int('SMARTFED_LOCAL_WEIGHT_MAX_BATCHES', 64)
    EVAL_BATCH_SIZE = 128
    EVAL_USE_AMP = True
    EVAL_MAX_BATCHES = _env_int('SMARTFED_EVAL_MAX_BATCHES', 64)
    EVAL_EVERY_N_ROUNDS = _env_int('SMARTFED_EVAL_EVERY_N_ROUNDS', 5)
    EVAL_OWN_DATASET_ONLY = False
    EVAL_FINAL_FULL = _env_str('SMARTFED_EVAL_FINAL_FULL', '1').lower() not in (
        '0', 'false', 'no',
    )

    # w_k in the KL term (R2-4 / Table VII). Default keeps the signed-loss APR update.
    WEIGHT_RULE = _env_str('SMARTFED_WEIGHT_RULE', 'signed_loss').lower()
    if WEIGHT_RULE not in ('uniform', 'dataset_size', 'signed_loss'):
        raise ValueError(
            f"Unsupported WEIGHT_RULE={WEIGHT_RULE!r}. "
            "Use 'uniform', 'dataset_size', or 'signed_loss'."
        )

    PUBLIC_KL_MAX_BATCHES = _env_int('SMARTFED_PUBLIC_KL_MAX_BATCHES', 60)
    PUBLIC_KL_LR = 3e-5
    PUBLIC_KL_TEMPERATURE = 3.0
    PUBLIC_KL_HEAD_ONLY = False
    PUBLIC_KL_SKIP_TEACHER0 = True
    PUBLIC_KL_FREEZE_TEACHER = True
    PUBLIC_KL_CONSENSUS = False
    PUBLIC_KL_USE_PUBLIC_MODEL = False

    GSAM_TOTAL_EPOCHS = _env_int('SMARTFED_GSAM_TOTAL_EPOCHS', 3)
    GSAM_MAX_BATCHES = _env_optional_int('SMARTFED_GSAM_MAX_BATCHES')
    GSAM_RHO_MAX = 0.05
    GSAM_RHO_MIN = 0.01
    GSAM_ALPHA = 0.5
    GSAM_LR = 0.01

    PARTICIPANT_PARAMS = {
        'loss_function': 'CE',
        'optimizer_name': 'Adam',
        'learning_rate': 0.001,
    }

    CLIENT_CONFIDENCE_REWEIGHT = True
    CLIENT_CONFIDENCE_REWEIGHT_LOSS = 'CE'
    BETA = 0.5 if CLIENT_CONFIDENCE_REWEIGHT else 0

    NOISE_TYPE = None
    NOISE_RATE = 0

    DATASET_TYPE = _env_str('SMARTFED_DATASET_TYPE', 'genimage').lower()

    if DATASET_TYPE == 'genimage':
        N_CLIENTS = 8
        DATASET_LIST = ['adm', 'BigGAN', 'glide', 'midjourney',
                        'sdv4', 'sdv5', 'vqdm', 'wukong']
        PRIVATE_DATASET_NAME = 'genimage'
        PRIVATE_DATASET_DIR = '/media/ethan/N-hub/mcsp_ubuntu/mcsp/FedAIDET/data/genimage_medium_fulltest'
        PRIVATE_DATA_LEN = 5000
        PRIVATE_DATASET_CLASSES = ['real', 'fake']
        OUTPUT_CHANNEL = len(PRIVATE_DATASET_CLASSES)
        PRIVATE_OUTPUT_CHANNEL = len(PRIVATE_DATASET_CLASSES)
        PUBLIC_DATASET_NAME = 'genimage'
        PUBLIC_DATASET_DIR = os.path.join(PRIVATE_DATASET_DIR, 'genimage_pub')
        PUBLIC_DATASET_LENGTH = 8000
        HAS_SERVER_TEACHER = True

    elif DATASET_TYPE == 'diffusionface':
        N_CLIENTS = 11
        DATASET_LIST = [
            'DDIM', 'DDPM', 'DiffSwap', 'LDM', 'P2', 'PNDM',
            'SDv1.5_T2I', 'SDv1.5_I2I', 'SDv2.1_T2I', 'SDv2.1_I2I', 'Inpaint',
        ]
        PRIVATE_DATASET_NAME = 'diffusionface'
        PRIVATE_DATASET_DIR = '/media/ethan/N-hub/mcsp_ubuntu/mcsp/FedAIDET/data/diffusionface'
        PRIVATE_DATA_LEN = 24000
        PRIVATE_DATASET_CLASSES = ['real', 'fake']
        OUTPUT_CHANNEL = len(PRIVATE_DATASET_CLASSES)
        PRIVATE_OUTPUT_CHANNEL = len(PRIVATE_DATASET_CLASSES)
        PUBLIC_DATASET_NAME = 'diffusionface'
        PUBLIC_DATASET_DIR = os.path.join(PRIVATE_DATASET_DIR, 'diffusionface_pub')
        PUBLIC_DATASET_LENGTH = 11000
        HAS_SERVER_TEACHER = True

    else:
        raise ValueError(
            f"Unsupported DATASET_TYPE={DATASET_TYPE!r} in config_genimage.py. "
            "For CIFAR-10 use SMARTFED_CONFIG=cifar (config_cifar.py)."
        )

    SERVER_TEACHER_ID = 0
    if HAS_SERVER_TEACHER:
        N_PARTIES = N_CLIENTS + 1
        CLIENT_IDS = list(range(1, N_PARTIES))
    else:
        N_PARTIES = N_CLIENTS
        CLIENT_IDS = list(range(N_CLIENTS))

    DCSA_MODE, AFWA_MODE, SELECTION_BUDGET = resolve_agent_modes(
        AGENT_MODE, DCSA_MODE, AFWA_MODE, N_CLIENTS, SELECTION_BUDGET
    )

    MODEL_STORAGE_ROOT = os.environ.get('SMARTFED_MODEL_STORAGE_ROOT', './Model_Storage')
    MODEL_STORAGE_MAIN_ROOT = os.environ.get(
        'SMARTFED_MODEL_STORAGE_MAIN_ROOT', './Model_Storage_main'
    )

    HETE_BACKBONE_POOL = [
        'EfficientNetB0',
        'ResNet10',
        'ResNet12',
        'Mobilenetv2',
    ]

    if PROTOCOL in ('Homo_FL', 'Homo_ServerGSAM'):
        PRIVATE_NETS_NAME_LIST = ['EfficientNetB0'] * N_PARTIES
    elif PROTOCOL == 'Hete_FL':
        client_nets = [
            HETE_BACKBONE_POOL[i % len(HETE_BACKBONE_POOL)]
            for i in range(N_CLIENTS)
        ]
        if HAS_SERVER_TEACHER:
            PRIVATE_NETS_NAME_LIST = ['EfficientNetB0'] + client_nets
        else:
            PRIVATE_NETS_NAME_LIST = client_nets
    else:
        raise ValueError(
            f"Unsupported PROTOCOL={PROTOCOL!r}. "
            "Use 'Homo_FL', 'Hete_FL', or 'Homo_ServerGSAM'."
        )

    NETS_NAME_LIST = list(PRIVATE_NETS_NAME_LIST)
    GSAM_UPDATE_SERVER = PROTOCOL == 'Homo_ServerGSAM'
