from typing import Optional

from pydantic import BaseModel


class SongCompatibility(BaseModel):
    timbre: int
    rhythm: int
    harmony: int
    mixable: int


class TransitionRate(BaseModel):
    transition_1: int
    transition_2: int
    transition_3: int
    transition_4: int
    transition_5: int
    transition_6: int
    transition_7: int


class SurveyResponse(BaseModel):
    question_rate_music_knowledge: str
    question_has_experience_djing: str

    question_song_compatability_autoMashUpper_0: SongCompatibility
    question_song_compatability_autoMashUpper_1: SongCompatibility
    question_song_compatability_autoMashUpper_2: SongCompatibility
    question_song_compatability_autoMashUpper_3: SongCompatibility
    question_song_compatability_autoMashUpper_4: SongCompatibility
    question_song_compatability_autoMashUpper_5: SongCompatibility
    question_song_compatability_autoMashUpper_6: SongCompatibility
    question_rate_transition_autoMashUpper: TransitionRate

    question_song_compatability_mosaikboxBare_0: SongCompatibility
    question_song_compatability_mosaikboxBare_1: SongCompatibility
    question_song_compatability_mosaikboxBare_2: SongCompatibility
    question_song_compatability_mosaikboxBare_3: SongCompatibility
    question_song_compatability_mosaikboxBare_4: SongCompatibility
    question_song_compatability_mosaikboxBare_5: SongCompatibility
    question_song_compatability_mosaikboxBare_6: SongCompatibility
    question_rate_transition_mosaikboxBare: TransitionRate

    question_rate_transition_mosaikboxWithStemRemoval: TransitionRate

    question_song_compatability_mosaikboxWithStemRemovalAndContextual_0: SongCompatibility
    question_song_compatability_mosaikboxWithStemRemovalAndContextual_1: SongCompatibility
    question_song_compatability_mosaikboxWithStemRemovalAndContextual_2: SongCompatibility
    question_song_compatability_mosaikboxWithStemRemovalAndContextual_3: SongCompatibility
    question_song_compatability_mosaikboxWithStemRemovalAndContextual_4: SongCompatibility
    question_song_compatability_mosaikboxWithStemRemovalAndContextual_5: SongCompatibility
    question_song_compatability_mosaikboxWithStemRemovalAndContextual_6: SongCompatibility
    question_rate_transition_mosaikboxWithStemRemovalAndContextual: TransitionRate
    question_email: Optional[str] = None
