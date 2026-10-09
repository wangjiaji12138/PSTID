"""Forecasting model registry. Baseline package paths remain compatible."""
from .registry import MODEL_LIST, MODEL_REGISTRY, build_model

__all__ = ["MODEL_LIST", "MODEL_REGISTRY", "build_model", "STSSL", "STSSDL"]


def __getattr__(name):
    if name in {"STSSL", "STSSDL"}:
        from importlib import import_module
        value = getattr(import_module(f"models.{name}.{name}"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
