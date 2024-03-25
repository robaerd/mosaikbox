from dataclasses import dataclass

import numpy as np


@dataclass
class Segment:
    start_time: float
    end_time: float
    beat_amount: int
    label: float
    compatibility: float  # mixability = compatibility * similarity
    timbre_vectors: np.ndarray | None = None
    start_beat_type: int | None = None  # 1 = downbeat, 2 = second beat, 3 = third beat, 4 = fourth beat
    end_beat_type: int | None = None
