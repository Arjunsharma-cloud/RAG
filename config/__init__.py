"""
Configuration module public API.
Exposes Settings model and load_config function.
"""

from .settings import Settings, load_config

__all__ = ["Settings", "load_config"]