from datetime import datetime

from mongoengine import Document, IntField, EmbeddedDocumentField, StringField, EmbeddedDocument, DateTimeField


class SongCompatibility(EmbeddedDocument):
    timbre = IntField(required=True)
    rhythm = IntField(required=True)
    harmony = IntField(required=True)
    mixable = IntField(required=True)


class TransitionRate(EmbeddedDocument):
    transition_1 = IntField(required=True)
    transition_2 = IntField(required=True)
    transition_3 = IntField(required=True)
    transition_4 = IntField(required=True)
    transition_5 = IntField(required=True)
    transition_6 = IntField(required=True)
    transition_7 = IntField(required=True)


class SurveyDocument(Document):
    question_rate_music_knowledge = StringField()
    question_has_experience_djing = StringField()
    # pageNo = IntField()
    question_song_compatability_autoMashUpper_0 = EmbeddedDocumentField(SongCompatibility)
    question_song_compatability_autoMashUpper_1 = EmbeddedDocumentField(SongCompatibility)
    question_song_compatability_autoMashUpper_2 = EmbeddedDocumentField(SongCompatibility)
    question_song_compatability_autoMashUpper_3 = EmbeddedDocumentField(SongCompatibility)
    question_song_compatability_autoMashUpper_4 = EmbeddedDocumentField(SongCompatibility)
    question_song_compatability_autoMashUpper_5 = EmbeddedDocumentField(SongCompatibility)
    question_song_compatability_autoMashUpper_6 = EmbeddedDocumentField(SongCompatibility)

    question_rate_transition_autoMashUpper = EmbeddedDocumentField(TransitionRate)
    question_song_compatability_mosaikboxBare_0 = EmbeddedDocumentField(SongCompatibility)
    question_song_compatability_mosaikboxBare_1 = EmbeddedDocumentField(SongCompatibility)
    question_song_compatability_mosaikboxBare_2 = EmbeddedDocumentField(SongCompatibility)
    question_song_compatability_mosaikboxBare_3 = EmbeddedDocumentField(SongCompatibility)
    question_song_compatability_mosaikboxBare_4 = EmbeddedDocumentField(SongCompatibility)
    question_song_compatability_mosaikboxBare_5 = EmbeddedDocumentField(SongCompatibility)
    question_song_compatability_mosaikboxBare_6 = EmbeddedDocumentField(SongCompatibility)

    question_rate_transition_mosaikboxBare = EmbeddedDocumentField(TransitionRate)

    question_rate_transition_mosaikboxWithStemRemoval = EmbeddedDocumentField(TransitionRate)
    question_song_compatability_mosaikboxWithStemRemovalAndContextual_0 = EmbeddedDocumentField(SongCompatibility)
    question_song_compatability_mosaikboxWithStemRemovalAndContextual_1 = EmbeddedDocumentField(SongCompatibility)
    question_song_compatability_mosaikboxWithStemRemovalAndContextual_2 = EmbeddedDocumentField(SongCompatibility)
    question_song_compatability_mosaikboxWithStemRemovalAndContextual_3 = EmbeddedDocumentField(SongCompatibility)
    question_song_compatability_mosaikboxWithStemRemovalAndContextual_4 = EmbeddedDocumentField(SongCompatibility)
    question_song_compatability_mosaikboxWithStemRemovalAndContextual_5 = EmbeddedDocumentField(SongCompatibility)
    question_song_compatability_mosaikboxWithStemRemovalAndContextual_6 = EmbeddedDocumentField(SongCompatibility)

    question_rate_transition_mosaikboxWithStemRemovalAndContextual = EmbeddedDocumentField(TransitionRate)
    question_email = StringField()
    # dates
    created_at = DateTimeField(default=datetime.utcnow)
