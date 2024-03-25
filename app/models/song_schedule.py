from dataclasses import dataclass
from typing import Optional

from app.models.mix import MixModel


@dataclass
class SongScheduleItem:
    audio_path: str
    mixability: Optional[float]
    start_time: float
    end_time: float
    total_play_time_start: Optional[float]
    total_play_time_end: Optional[float]
    original_tempo: float
    target_tempo: Optional[float]
    original_key: str
    target_key: str
    key_shift: int
    h_contr: float
    l_contr: float
    r_contr: float
    timbral_contr: float
    segment_factor: float
    vocal_intervals_time_stretched: Optional[list[tuple[int, int]]]
    mix_model: MixModel

    def serialize(self):
        return {
            "audio_path": self.audio_path,
            "mixability": self.mixability,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_play_time_start": self.total_play_time_start,
            "total_play_time_end": self.total_play_time_end,
            "original_tempo": self.original_tempo,
            "target_tempo": self.target_tempo,
            "original_key": self.original_key,
            "target_key": self.target_key,
            "key_shift": self.key_shift,
            "h_contr": self.h_contr,
            "l_contr": self.l_contr,
            "r_contr": self.r_contr,
            "timbral_contr": self.timbral_contr,
            "segment_factor": self.segment_factor,
            "vocal_intervals_time_stretched": self.vocal_intervals_time_stretched,
            "mix_model": self.mix_model.value
        }
