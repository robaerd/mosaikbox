from enum import Enum

from pydantic import BaseModel


class AnalysisState(Enum):
    UPLOAD = "UPLOAD"
    STARTED = "STARTED"
    LYRICS_DOWNLOADING = "LYRICS_DOWNLOADING"
    MSS = "MSS"
    ANALYSIS = "ANALYSIS"
    SCHEDULING = "SCHEDULING"
    WAITING_FOR_MIXING = "WAITING_FOR_MIXING"
    MIXING = "MIXING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class AnalysisStatus(BaseModel):
    state: AnalysisState
    error_message: str | None = None
