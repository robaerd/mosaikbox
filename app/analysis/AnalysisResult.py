from dataclasses import dataclass
from typing import Optional

import numpy as np
from torch import Tensor

from .Segment import Segment
from ..models.mix import MixModel


@dataclass
class AnalysisResult:
    audio_path: str
    filename_with_hash: str
    bpm: float
    target_bpm: Optional[float]
    key: str
    target_key: Optional[str]
    beats: np.ndarray
    quantized_percussion: np.ndarray
    downbeat_segments: list[Segment]
    sample_rate: int
    audio_length: float
    time_until_first_beat_stretched: Optional[float]
    stretch_playback_rate: Optional[float]
    lyrics_embedding: Optional[Tensor]
    vocal_segments: Optional[list[tuple[int, int]]]
    mix_model: MixModel
