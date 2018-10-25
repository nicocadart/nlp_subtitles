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

PERSONS = ['Howard', 'Raj', 'Sheldon', 'Penny', 'Leonard']


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

