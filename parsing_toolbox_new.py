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


def load_db():
    with open(DATABASE_PATH, "r", newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='§')
        return list(reader)
