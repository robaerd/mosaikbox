from enum import Enum

from pydantic import BaseModel


class MixModel(Enum):
    MOSAIKBOX = "MOSAIKBOX"
    AMU = "AMU"


class MixInitiateResponse(BaseModel):
    session_id: str


class AnalysisStartRequest(BaseModel):
    start_song_name: str
    is_contextual_information_enabled: bool
    mix_model: MixModel


class MixStartRequest(BaseModel):
    is_drum_removal_enabled: bool
    is_vocal_removal_enabled: bool
    mix_model: MixModel
