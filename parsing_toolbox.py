import numpy as np


SCENE = 0
EPISODE = 1
SEASON = 2
TIME = 3
DURATION = 4
WORD = 5
PERSON = 8

DB_DIR = 'data'


def load_episode_db(episode):
    """
    Load numpy database corresponding to episode "episode".
    :param episode: int, number of episode, between 1 and 12
    :return: numpy array of episode
    """
    return np.load("{}/db_words_S1E{}.npy".format(DB_DIR, int(episode)))


def get_scene_text(db, scene):
    """
    Return raw text of a full scene.
    :param db: the numpy database of the episode
    :param scene: int, the number of the scene
    :return: string, full text of the scene
    """
    return " ".join([word[WORD] for word in db if float(word[0]) == scene])


def get_scene(db, scene):
    """
    Return part of the episode database corresponding to a given scene.
    :param db: the numpy database of the episode
    :param scene: int, the number of the scene
    :return: part of the numpy database corresponding to the scene
    """
    return db[db[:, 0].astype(np.float) == scene]


def get_scenes_numbers(db):
    """
    Get the numbers of scenes in the given epidose database.
    :param db: the numpy database of the episode
    :return: list, the scenes numbers of the given db
    """
    return np.sort(np.unique(db[:, SCENE]).astype(np.float))
