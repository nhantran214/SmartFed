"""Shared Config helpers for GenImage (`config_genimage.py`) and CIFAR (`config_cifar.py`)."""

import os

_SMARTFED_ROOT = os.path.dirname(os.path.abspath(__file__))


def _env_str(key, default):
    val = os.environ.get(key)
    return default if val is None or val == '' else val


def _env_int(key, default):
    val = os.environ.get(key)
    if val is None or val == '':
        return default
    return int(val)


def _env_optional_int(key):
    val = os.environ.get(key)
    if val is None or val == '':
        return None
    return int(val)


def resolve_agent_modes(agent_mode, dcsa_mode, afwa_mode, n_clients, selection_budget):
    """Fill DCSA_MODE / AFWA_MODE / SELECTION_BUDGET from AGENT_MODE."""
    agent_defaults = {
        'wo_ma': ('off', 'off'),
        'without_ma': ('off', 'off'),
        'off': ('off', 'off'),
        'rule_based': ('rule_based', 'rule_based'),
        'rule': ('rule_based', 'rule_based'),
        'llm': ('llm', 'llm'),
        'custom': ('llm', 'llm'),
    }
    d_default, a_default = agent_defaults.get(agent_mode, ('llm', 'llm'))
    if not dcsa_mode:
        dcsa_mode = d_default
    if not afwa_mode:
        afwa_mode = a_default
    if dcsa_mode in ('wo_ma', 'without_ma'):
        dcsa_mode = 'off'
    if afwa_mode in ('wo_ma', 'without_ma'):
        afwa_mode = 'off'
    if dcsa_mode not in ('off', 'rule_based', 'llm'):
        raise ValueError(f'Unsupported DCSA_MODE={dcsa_mode!r}')
    if afwa_mode not in ('off', 'rule_based', 'llm'):
        raise ValueError(f'Unsupported AFWA_MODE={afwa_mode!r}')
    if selection_budget is None:
        selection_budget = n_clients
    return dcsa_mode, afwa_mode, int(selection_budget)


class ConfigPartyMixin:
    """Party-index helpers. Subclasses must set CLIENT_IDS / DATASET_LIST / …"""

    @classmethod
    def is_server_teacher(cls, party_id):
        return bool(getattr(cls, 'HAS_SERVER_TEACHER', False)) and int(party_id) == int(
            getattr(cls, 'SERVER_TEACHER_ID', 0)
        )

    @classmethod
    def gsam_party_ids(cls):
        ids = list(cls.CLIENT_IDS)
        if getattr(cls, 'GSAM_UPDATE_SERVER', False) and getattr(
            cls, 'HAS_SERVER_TEACHER', False
        ):
            tid = int(cls.SERVER_TEACHER_ID)
            if tid not in ids:
                ids = [tid] + ids
        return ids

    @classmethod
    def client_local_index(cls, party_id):
        return cls.CLIENT_IDS.index(int(party_id))

    @classmethod
    def party_dataset_key(cls, party_id):
        if cls.is_server_teacher(party_id):
            return os.path.basename(cls.PUBLIC_DATASET_DIR.rstrip(os.sep)) or 'public'
        return cls.DATASET_LIST[cls.client_local_index(party_id)]

    @classmethod
    def party_datadir(cls, party_id):
        if cls.is_server_teacher(party_id):
            return cls.PUBLIC_DATASET_DIR
        return os.path.join(
            cls.PRIVATE_DATASET_DIR,
            cls.DATASET_LIST[cls.client_local_index(party_id)],
        )
