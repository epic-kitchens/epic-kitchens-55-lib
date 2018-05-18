"""Column names present in a labels dataframe.

    Rather than accessing column names directly, we suggest you import these constants
    and use them to access the data in case the names change at any point.
"""


"""UID column name, a number uniquely identifying a particular action segment

    e.g. ``6374``"""
UID_COL = "uid"

"""Video column name, an identifier for a specific video of the form P\\d\\d_\\d\\d,\
    the first two digits are the participant ID, and the last two digits the video ID

    e.g. ``\"P03_01\"``"""
VIDEO_ID_COL = "video_id"

"""Narration column name, the original narration by the participant about the action performed

    e.g. ``\"close fridge\"``"""
NARRATION_COL = "narration"

"""Start timestamp column name, the timestamp of the start of the action segment

    e.g. ``\"00:23:43.847\"``"""
START_TS_COL = "start_timestamp"

"""Stop timestamp column name, the timestamp of the end of the action segment

    e.g. ``\"00:23:47.212\"``"""
STOP_TS_COL = "stop_timestamp"


"""Start frame column name, the frame corresponding to the starting timestamp

    e.g. ``85430``"""
START_F_COL = "start_frame"


"""Stop frame column name, the frame corresponding to the starting timestamp

    e.g. ``85643``"""
STOP_F_COL = "stop_frame"


"""Participant ID column name, the identifier corresponding to an individual

    e.g. ``85643``"""
PARTICIPANT_ID_COL = "participant_id"

"""Verb column name, the first verb extracted from the narration

    e.g. ``\"close\"``"""
VERB_COL = "verb"

"""Verb class column name, the class corresponding to the verb extracted from the narration.

    e.g. ``3``"""
VERB_CLASS_COL = "verb_class"

"""Noun column name, the first noun extracted from the narration

    e.g. ``\"fridge\"``"""
NOUN_COL = "noun"

"""Noun class column name, the class corresponding to the first noun extracted from the narration

    e.g. ``10``"""
NOUN_CLASS_COL = "noun_class"


"""Nouns column name, all nouns extracted from the narration

    e.g. ``[\"fridge\"]``"""
NOUNS_COL = "all_nouns"

"""Nouns class column name, the classes corresponding to each noun extracted from the narration

    e.g. ``[10]``"""
NOUNS_CLASS_COL = "all_noun_classes"


"""The noun class corresponding to an action without a noun, consider the narration \"stir\" where
    no object is specified."""
EMPTY_NOUN_CLASS = 0
