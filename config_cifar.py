"""CIFAR-10 Config for SmartFed (R3-Major-4 / Figs. 4–5).

Activate with:
  SMARTFED_CONFIG=cifar python main.py
  SMARTFED_DATASET_TYPE=cifar10 python main.py

Defaults (fixes vs the first CIFAR ablation run):
  - 16 clients, Homo_FL, all clients GSAM unless Fig.4 sets SELECTION_BUDGET=4
  - GSAM_TOTAL_EPOCHS=5
  - Public KL: head-only, 20 batches, T=2.0, lr=5e-5
  - KL teacher = public_model.ckpt (not client 0)
  - AFWA coefficients clipped to [0.05, 0.3]
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
    PROTOCOL = _env_str('SMARTFED_PROTOCOL', 'Homo_FL')
    SEED = _env_int('SMARTFED_SEED', 0)
    N_CLIENTS = _env_int('SMARTFED_N_CLIENTS', 16)

    TRAIN_BATCH_SIZE = 32
    TEST_BATCH_SIZE = 32
    TRAIN_BATCH_SIZE_PRIVATE = 32
    TEST_BATCH_SIZE_PRIVATE = 32
    COMMUNICATION_EPOCH = _env_int('SMARTFED_COMMUNICATION_EPOCH', 40)
    PRETRAIN_EPOCH = _env_int('SMARTFED_PRETRAIN_EPOCH', 100)
    INTRA_DATASET = False
    PRETRAIN_PARTY_IDS = None

    AGENT_MODE = _env_str('SMARTFED_AGENT_MODE', 'llm').lower()
    DCSA_MODE = _env_str('SMARTFED_DCSA_MODE', '').lower()
    AFWA_MODE = _env_str('SMARTFED_AFWA_MODE', '').lower()
    DCSA_RULE_STRATEGY = _env_str('SMARTFED_DCSA_RULE_STRATEGY', 'adaptive').lower()
    # Agent-chain default: all 16 clients. Fig.4 only: SMARTFED_SELECTION_BUDGET=4
    SELECTION_BUDGET = _env_optional_int('SMARTFED_SELECTION_BUDGET')
    AFWA_FIXED_COEFFICIENT = float(_env_str('SMARTFED_AFWA_FIXED_COEFFICIENT', '0.1'))
    AFWA_COEFF_MIN = float(_env_str('SMARTFED_AFWA_COEFF_MIN', '0.05'))
    AFWA_COEFF_MAX = float(_env_str('SMARTFED_AFWA_COEFF_MAX', '0.3'))
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

    LOCAL_WEIGHT_MAX_BATCHES = 64
    EVAL_BATCH_SIZE = 128
    EVAL_USE_AMP = True
    EVAL_MAX_BATCHES = 64
    EVAL_EVERY_N_ROUNDS = 5
    EVAL_OWN_DATASET_ONLY = False
    EVAL_FINAL_FULL = True

    PUBLIC_KL_MAX_BATCHES = 20
    PUBLIC_KL_LR = 5e-5
    PUBLIC_KL_TEMPERATURE = 2.0
    PUBLIC_KL_HEAD_ONLY = True
    PUBLIC_KL_SKIP_TEACHER0 = True
    PUBLIC_KL_FREEZE_TEACHER = True
    PUBLIC_KL_CONSENSUS = False
    # CIFAR has no party-0 server teacher; distill from the public split model.
    PUBLIC_KL_USE_PUBLIC_MODEL = True

    GSAM_TOTAL_EPOCHS = 10
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

    DATASET_TYPE = 'cifar10'
    DATASET_LIST = ['cifar10'] * N_CLIENTS
    PRIVATE_DATASET_NAME = 'cifar10'
    PRIVATE_DATASET_DIR = _env_str(
        'SMARTFED_PRIVATE_DATASET_DIR',
        os.path.join(_SMARTFED_ROOT, 'data', 'cifar'),
    )
    CIFAR10_PICKLE_DIR = _env_str(
        'SMARTFED_CIFAR10_PICKLE_DIR',
        os.path.join(_SMARTFED_ROOT, 'data', 'cifar-10-python'),
    )
    PRIVATE_DATA_LEN = 5000
    # Public hold-out before splitting the rest across N_CLIENTS (IID).
    CIFAR_PUBLIC_TRAIN = _env_int('SMARTFED_CIFAR_PUBLIC_TRAIN', 5000)
    CIFAR_PUBLIC_TEST = _env_int('SMARTFED_CIFAR_PUBLIC_TEST', 1000)
    PRIVATE_DATASET_CLASSES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck',
    ]
    OUTPUT_CHANNEL = len(PRIVATE_DATASET_CLASSES)
    PRIVATE_OUTPUT_CHANNEL = len(PRIVATE_DATASET_CLASSES)
    PUBLIC_DATASET_NAME = 'cifar10'
    PUBLIC_DATASET_DIR = _env_str('SMARTFED_PUBLIC_DATASET_DIR', PRIVATE_DATASET_DIR)
    PUBLIC_DATASET_LENGTH = CIFAR_PUBLIC_TRAIN
    HAS_SERVER_TEACHER = False

    SERVER_TEACHER_ID = 0
    N_PARTIES = N_CLIENTS
    CLIENT_IDS = list(range(N_CLIENTS))

    DCSA_MODE, AFWA_MODE, SELECTION_BUDGET = resolve_agent_modes(
        AGENT_MODE, DCSA_MODE, AFWA_MODE, N_CLIENTS, SELECTION_BUDGET
    )

    MODEL_STORAGE_ROOT = os.environ.get(
        'SMARTFED_MODEL_STORAGE_ROOT', './Model_Storage_cifar10'
    )
    MODEL_STORAGE_MAIN_ROOT = os.environ.get(
        'SMARTFED_MODEL_STORAGE_MAIN_ROOT', './Model_Storage_main_cifar10'
    )

    HETE_BACKBONE_POOL = [
        'EfficientNetB0',
        'ResNet10',
        'ResNet12',
        'Mobilenetv2',
    ]

    HOMO_BACKBONE = _env_str('SMARTFED_HOMO_BACKBONE', 'EfficientNetB0')
    _bb = HOMO_BACKBONE.strip().lower().replace('-', '').replace('_', '')
    if _bb in ('efficientnetb0', 'b0'):
        HOMO_BACKBONE = 'EfficientNetB0'
    elif _bb in ('efficientnetb4', 'b4'):
        HOMO_BACKBONE = 'EfficientNetB4'
    if PROTOCOL in ('Homo_FL', 'Homo_ServerGSAM'):
        PRIVATE_NETS_NAME_LIST = [HOMO_BACKBONE] * N_PARTIES
    elif PROTOCOL == 'Hete_FL':
        PRIVATE_NETS_NAME_LIST = [
            HETE_BACKBONE_POOL[i % len(HETE_BACKBONE_POOL)]
            for i in range(N_CLIENTS)
        ]
    else:
        raise ValueError(
            f"Unsupported PROTOCOL={PROTOCOL!r}. "
            "Use 'Homo_FL', 'Hete_FL', or 'Homo_ServerGSAM'."
        )

    NETS_NAME_LIST = list(PRIVATE_NETS_NAME_LIST)
    GSAM_UPDATE_SERVER = False
