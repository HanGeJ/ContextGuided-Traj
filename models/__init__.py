"""Paper models: fixed-budget trajectory/social and optional visual refinement."""
from .factory import VARIANT_TO_MODEL, build_model
from .visual_context import build_visual_model
__all__ = ["VARIANT_TO_MODEL", "build_model", "build_visual_model"]
