"""Setting definitions for model zoo."""

from .aesthetic_preferences import build_aesthetic_preferences_setting
from .emergent_misalignment import build_emergent_misalignment_setting
from .insecure_code import build_insecure_code_setting
from .simple_math import build_simple_math_setting

__all__ = [
    "build_aesthetic_preferences_setting",
    "build_emergent_misalignment_setting",
    "build_insecure_code_setting",
    "build_simple_math_setting",
]
