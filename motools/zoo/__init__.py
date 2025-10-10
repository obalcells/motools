"""Zoo module for curated specimens and registry."""

from .specimen import Specimen, get_specimen, list_specimens, register_specimen

__all__ = ["Specimen", "register_specimen", "get_specimen", "list_specimens"]
