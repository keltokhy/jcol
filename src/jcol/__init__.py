"""Apply natural-language codebooks to tables, with resumable annotation and evaluation."""

__version__ = "0.3.1"

from .batch import AnnotationResult, annotate, annotate_async
from .codebook import Codebook
from .evaluation import evaluate

__all__ = ["AnnotationResult", "Codebook", "annotate", "annotate_async", "evaluate"]
