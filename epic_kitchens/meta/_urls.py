LABEL_VERSION = "v1.5.0"

_url_root = "https://github.com/epic-kitchens/annotations/raw/{}/".format(LABEL_VERSION)

verb_classes_url = _url_root + "EPIC_verb_classes.csv"
noun_classes_url = _url_root + "EPIC_noun_classes.csv"

training_labels_url = _url_root + "EPIC_train_action_labels.pkl"
training_object_labels_url = _url_root + "EPIC_train_object_labels.csv"
training_narrations_url = _url_root + "EPIC_train_action_narrations.csv"

test_seen_timestamps_url = _url_root + "EPIC_test_s1_timestamps.pkl"
test_unseen_timestamps_url = _url_root + "EPIC_test_s2_timestamps.pkl"

many_shot_actions_url = _url_root + "EPIC_many_shot_actions.csv"
many_shot_verbs_url = _url_root + "EPIC_many_shot_verbs.csv"
many_shot_nouns_url = _url_root + "EPIC_many_shot_nouns.csv"

descriptions_url = _url_root + "EPIC_descriptions.csv"
video_info_url = _url_root + "EPIC_video_info.csv"
