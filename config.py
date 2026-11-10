"""SmartFed configuration dispatcher.

Default (GenImage / DiffusionFace): `config_genimage.py`
CIFAR-10: `config_cifar.py` when
  SMARTFED_CONFIG=cifar   or   SMARTFED_DATASET_TYPE=cifar10
"""

import os

from config_base import _SMARTFED_ROOT, _env_int, _env_optional_int, _env_str  # noqa: F401


def _use_cifar_config():
    cfg = os.environ.get('SMARTFED_CONFIG', '').lower()
    ds = os.environ.get('SMARTFED_DATASET_TYPE', '').lower()
    return cfg in ('cifar', 'cifar10') or ds == 'cifar10'


if _use_cifar_config():
    from config_cifar import Config
else:
    from config_genimage import Config

__all__ = ['Config', '_SMARTFED_ROOT']
