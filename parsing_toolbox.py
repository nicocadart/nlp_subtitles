import numpy as np
from nltk import sent_tokenize
from encoding import one_hot_encoding

SCENE = 0
EPISODE = 1
SEASON = 2
TIME = 3
DURATION = 4
WORD = 5
PERSON = 8

DB_DIR = 'data'
PERSONS = ['howard_wolowitz', 'sheldon_cooper', 'leonard_hofstadter', 'penny',
           'rajesh_koothrappali']


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


def words_to_sentences(words):
    """
    @brief: takes a text as a list of words (or 1d ndarray) and return a list of sentences,
            as well as the index of the corresponding sentences for each word (in 1d array)
    @param:
        words: list (or 1d array) of string, the list of words we want to sentence-tokenize

    @return:
        list_sentences: list of list, text divided in sentences (each list being a sentence)
        idx_word2sentences: 1d ndarray int of size n_words,
                                index of the list associated to each word of words
        idx_sentences2first_word: 1d ndarray int of size n_sentences,
                                    index in words of the word of each sentence

    """
    n_words = len(words)
    text = ' '.join(words)
    list_sentences = list(sent_tokenize(text))

    current_idx_word = 0
    idx_word2sentences = np.zeros((n_words,))
    idx_sentences2first_word = np.zeros((len(list_sentences,)))
    for (i, sentence) in enumerate(list_sentences):
        idx_sentences2first_word[i] = current_idx_word
        # We split again our sentence
        words_in_sentence = sentence.split(' ')

        # for each word of the current sentence, match the correct word in words list and increment
        # the index of the words we're tagging
        for (j, w) in enumerate(words_in_sentence):
            # Sanity check, should always pass
            if w == words[current_idx_word]:
                idx_word2sentences[current_idx_word] = i
                current_idx_word += 1
            else:
                print('error:', i, j, current_idx_word)

    return list_sentences, idx_word2sentences.astype(int), idx_sentences2first_word.astype(int)

def load_sentences_persons(list_ep, states=PERSONS):
    """
    @brief: load a list of episodes files, and create a database of the sentences,
                associated with the labels being the person who said it
    @param:

    @return:

    """
    all_sentences_in_corpus, labels = [], []
    for ep in list_ep:
        db = load_episode_db(ep)
        sentences, _, idx_s2w = words_to_sentences(list(db[:, WORD]))
        label_sentences = np.array([db[i, PERSON] for i in idx_s2w])
        label_sentences = np.argmax(one_hot_encoding(label_sentences, states), axis=1)

        all_sentences_in_corpus += sentences
        labels += list(label_sentences)

    return np.array(all_sentences_in_corpus), np.array(labels)
