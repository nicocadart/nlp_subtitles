import numpy as np
import csv
from nltk import sent_tokenize
from encoding import one_hot_encoding

SEASON = 0
EPISODE = 1
SCENE = 2
SCENE_ID = 3
PERSON = 4
SENTENCE = 5

DATABASE_PATH  = 'data/tbbt_db.csv'
DB_COLUMNS = ['season', 'episode', 'scene', 'scene_id', 'word', 'locutor']

PERSONS = ['Sheldon', 'Leonard', 'Penny', 'Raj', 'Howard']
UNKNOWN_STATE = ['Unknown']

def load_db():
    with open(DATABASE_PATH, "r", newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='§')
        return list(reader)

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
    state_space = PERSONS + UNKNOWN_STATE
    # Loading the list of sentences
    db = load_db()

    for sentence in db:
        sentences.append(sentence[SENTENCE])
        scenes_id.append(sentence[SCENE_ID])

        # Check if the person is an accepted state
        person = sentence[PERSON]
        if person not in states:
            person = UNKNOWN_STATE[0]

        label = state_space.index(person)
        labels.append(label)

    return sentences, scenes_id, labels
