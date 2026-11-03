"""Versioned upstream configuration, independent of training CLI defaults.

The JSON separates upstream evidence from settings supported by our adapters.
Baseline runtime settings are frozen in registry.py; this module stores provenance.
"""
import json
from copy import deepcopy
from pathlib import Path


MODEL_CONFIG_REGISTRY = json.loads(
    Path(__file__).with_name('official_configs.json').read_text(encoding='utf-8')
)


def get_model_config(name):
    """Return an isolated, JSON-serializable configuration and provenance record."""
    return deepcopy(MODEL_CONFIG_REGISTRY[name.upper()])


def model_defaults(name):
    config = get_model_config(name)
    return {**config['model'], **config['training']}


def resolve_model_args(name, args):
    """Compatibility wrapper; the registry owns the baseline locking policy."""
    from .registry import resolve_model_args as resolve
    return resolve(name, args)
