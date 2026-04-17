"""Part 1: Speech-to-Text with Language Identification"""

from .transcription_pipeline import TranscriptionPipeline
from .lid_model import MultiHeadLID

__all__ = ['TranscriptionPipeline', 'MultiHeadLID']
