import numpy as np
import csv

SEASON = 0
EPISODE = 1
SCENE = 2
SCENE_ID = 3
PERSON = 4
SENTENCE = 5

DATABASE_PATH  = 'data/tbbt_db.csv'
DB_COLUMNS = ['season', 'episode', 'scene', 'scene_id', 'locutor', 'text']
DELIMITER = '§'

PERSONS = ['Sheldon', 'Leonard', 'Penny', 'Raj', 'Howard']
UNKNOWN_STATE = 'Unknown'


def load_db():
    """
    Load the full database.
    :return: list of list, where each element has format ['season', 'episode', 'scene', 'scene_id', 'locutor', 'text']
    """
    with open(DATABASE_PATH, "r", newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=DELIMITER)
        return list(reader)


def get_persons_scenes(db):
    """
    Parse database and return persons and text for each scene.
    :param db: the full database, returned by load_db()
    :return: persons: list of list, where each element is a person talking during the given scene
    :return: text: list of strings, where each element is the full text of the given scene
    """
    persons = []
    text = []

    current_scene_id = 'x'

    current_scene_persons = set()
    current_scene_text = ""

    for db_entry in db:

        if db_entry[SCENE_ID] != current_scene_id:
            if current_scene_id != "x":
                persons.append(list(current_scene_persons))
                text.append(current_scene_text)
            current_scene_id = db_entry[SCENE_ID]
            current_scene_persons = set()
            current_scene_text = ""

        current_scene_persons.add(db_entry[PERSON])
        current_scene_text += db_entry[SENTENCE]

    return persons, text


def load_sentences_by_person(states=PERSONS):
    """
    @brief: load a list of episodes files, and create a database of the sentences,
                associated with the labels being the person who said it
    @param:
            states: list of string, with the accepted names for the labels,
                    any others will be marked as UNKNOWN_STATE

    @return:

    """
    sentences, scenes_id, labels = [], [], []
    state_space = PERSONS + [UNKNOWN_STATE]
    # Loading the list of sentences
    db = load_db()

    for sentence in db:
        sentences.append(sentence[SENTENCE])
        scenes_id.append(sentence[SCENE_ID])

        # Check if the person is an accepted state
        person = sentence[PERSON]
        if person not in states:
            person = UNKNOWN_STATE

        label = state_space.index(person)
        labels.append(label)

    return sentences, scenes_id, labels
