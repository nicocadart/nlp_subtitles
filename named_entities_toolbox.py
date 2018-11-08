from os.path import isfile
import csv
from collections import Counter
import numpy as np

import nltk

from parsing_toolbox import load_db, get_persons_scenes, PERSONS, UNKNOWN_STATE
from encoding import one_hot_encoding


INDEX_SETS_PATH = "data/train_test_split_scenes_indices.npy"
PERSONS_NE_DB = "data/persons_ne_db.csv"


def get_named_entities(tokens, category=None, tokenize=False, once=True):
    """
    Return named entities found in some text.
    :param tokens: array or list of words. If a full text is given (without words being segmented, use tokenize=True)
    :param category: if provided, return only the list of named entities of type 'category'
    :param tokenize: needs to be set to True if 'tokens' is a raw text given as input.
    :param once: if True, each named entities is returned only once (remove several occurences)
    :return: if category=None: a dic with keys as detected named entities types, and values as list of detections.
             else: a list of detections of type 'category'
    """
    # tokenize raw text if necessary
    if tokenize:
        tokens = nltk.word_tokenize(tokens)

    # get named entities
    ne_chunk_tree = nltk.ne_chunk(nltk.pos_tag(tokens))
    ne_dict = {chunk.label(): [] for chunk in ne_chunk_tree if hasattr(chunk, 'label')}
    for chunk in ne_chunk_tree:
        if hasattr(chunk, 'label'):
            ne_dict[chunk.label()] += [c[0] for c in chunk]

    # filter named entities
    if once:
        for ne in ne_dict.keys():
            ne_dict[ne] = list(set(ne_dict[ne]))

    # return named entities
    if category:
        return ne_dict[category] if category in ne_dict.keys() else []
    else:
        return ne_dict


def build_ne_persons_dataset():
    """
    @brief: build a list of episodes files, and create a database of the named entities in each scene,
                associated with the labels being the person who said it
    @return: named_entities : list of found named entities in each scene
             labels: list of locutors in each scene
    """
    named_entities = []
    db = load_db()
    scenes_persons, scenes_text, scene_ids = get_persons_scenes(db)
    for persons, text in zip(scenes_persons, scenes_text):

        # get named entities
        ne_dict = get_named_entities(text, once=False, tokenize=True)
        ne = [word for words in ne_dict.values() for word in words]

        # update dataset
        named_entities.append(ne)

    return named_entities, scenes_persons, scene_ids


def load_ne_persons_dataset():
    """
    Load full named_entities and locutors per scene dataset, either by reading existing file or building it.
    :return: named_entities : list of found named entities in each scene
             persons: list of locutors in each scene
    """
    # read database
    named_entities, persons, scene_ids = [], [], []

    # Load dataset if already exists
    if isfile(PERSONS_NE_DB):
        print("Loading named entities dataset from pre-built file \'{}\'".format(PERSONS_NE_DB))
        with open(PERSONS_NE_DB, "r", newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='§')
            for row in reader:
                scene_id = row[0]
                locutors = row[1].split("|")
                ne = row[2].split("|")
                scene_ids.append(scene_id)
                named_entities.append(ne)
                persons.append(locutors)

    else:
        # build dataset
        print("Building named entities dataset")
        named_entities, persons, scene_ids = build_ne_persons_dataset()

        # write dataset for future executions
        print("Saving named entities dataset to file \'{}\'".format(PERSONS_NE_DB))
        with open(PERSONS_NE_DB, "w", newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter='§')
            for scene_id, ne, locutors in zip(scene_ids, named_entities, persons):
                ne_str = "|".join(ne)
                locutors_str = "|".join(locutors)
                writer.writerow([scene_id, locutors_str, ne_str])

    return named_entities, persons, scene_ids


def clean_ne_persons_dataset(named_entities, persons, min_ne_count=5, states=PERSONS, unknown_state=UNKNOWN_STATE, once=False):
    """
    Filter data by selecting only named entities occuring more than "min_ne_count", and replacing all states which are
    not in "states" by "unknown_state". If "once" is set to True, only one occurence of each NE in each scene will be
    kept.
    """
    ne_count = Counter([ne for ne_scene in named_entities for ne in ne_scene])
    ne_vocab = {ne for ne, count in ne_count.items() if count >= min_ne_count}

    ne_cleaned, persons_cleaned = [], []
    for i_scene in range(len(named_entities)):

        # clean labels
        locutors = list({person if person in states else unknown_state for person in persons[i_scene]})

        # clean named entities
        if once:
            ne = list({ne for ne in named_entities[i_scene] if ne in ne_vocab})
        else:
            ne = [ne for ne in named_entities[i_scene] if ne in ne_vocab]

        ne_cleaned.append(ne)
        persons_cleaned.append(locutors)

    return ne_cleaned, persons_cleaned


def split_train_test_ne_persons_dataset(named_entities, persons, scene_ids, train_val_test_path=INDEX_SETS_PATH, possible_locutors=PERSONS):
    """
    Convert ne/persons database into a trainable hot encoding database split into train, validation and test sets.
    """
    # constants
    n_samples = len(named_entities)
    scene_ids = np.array(scene_ids)
    persons = np.array(persons)
    indexes = np.load(train_val_test_path)
    train_ids, valid_ids, test_ids = indexes[0], indexes[1], indexes[2]

    # build X : hot encoded named_entities
    ne_vocab = {ne for ne_scene in named_entities for ne in ne_scene if ne}
    n_features = len(ne_vocab)
    X = np.zeros((n_samples, n_features))
    for i_scene in range(n_samples):
        X[i_scene, :] = np.sum(one_hot_encoding(named_entities[i_scene], list(ne_vocab), unknown_name=''), axis=0)
    X_train = X[np.isin(scene_ids, train_ids), :]
    X_valid = X[np.isin(scene_ids, valid_ids), :]
    X_test = X[np.isin(scene_ids, test_ids), :]

    # build y : one hot encoded persons
    y_train, y_valid, y_test = {}, {}, {}
    for person in possible_locutors:
        y_train[person] = np.array([1 if person in locutors else 0 for locutors in persons[np.isin(scene_ids, train_ids)]])
        y_valid[person] = np.array([1 if person in locutors else 0 for locutors in persons[np.isin(scene_ids, valid_ids)]])
        y_test[person]  = np.array([1 if person in locutors else 0 for locutors in persons[np.isin(scene_ids, test_ids)]])

    return X_train, y_train, train_ids, X_valid, y_valid, valid_ids, X_test, y_test, test_ids


def get_train_test_ne_persons_dataset(possible_locutors, ne_min_count=10, once=False):

    # Load dataset
    named_entities_full, persons_full, scene_ids = load_ne_persons_dataset()

    # Clean dataset :
    #   - replace all occurences of unkown characters by UNKOWN_STATE
    #   - remove named_entities counted less than 'min_count' times
    #   - keep only a single occurence of each NE in each scene if "once" is True
    print("Cleaning dataset with min_ne_count = {}".format(ne_min_count))
    named_entities, persons = clean_ne_persons_dataset(named_entities_full,
                                                       persons_full,
                                                       min_ne_count=ne_min_count,
                                                       once=once)

    # # Display remaining dataset
    # for ne, pers in zip(named_entities_cleaned, persons_cleaned):
    #     print("{:<85}: {}".format(" ".join(pers), " ".join(ne)))

    # Get train and test datasets
    X_train, y_train, ids_train, X_valid, y_valid, ids_valid, X_test, y_test, ids_test = split_train_test_ne_persons_dataset(named_entities,
                                                                                                                           persons,
                                                                                                                           scene_ids,
                                                                                                                           possible_locutors=possible_locutors)

    return X_train, y_train, ids_train, X_valid, y_valid, ids_valid, X_test, y_test, ids_test
