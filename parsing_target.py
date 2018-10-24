from os import listdir
from os.path import isfile, join
import numpy as np
from nltk import sent_tokenize

# SEASON EPISODE SCENE SENTENCE WORD TIME DURATION type_EN nom_EN role_EN nom_locuteur

def one_hot_encoding(labels, states):
    """@brief: Return one_hot encoding of an array of labels for a given state space of strings
        If a label isn't in the state space, it is labelled "Unknown"

        @param
        labels: ndarray, of size (n_samples, )
        states: list of strings of len n_states, each string being a state of your system

        @return:
        one_hot_encode: ndarray, of size (n_samples, n_states + 1)
    """
    # Adding the "unknown" state to the state space
    states = states + ['unknown']
    n_samples = labels.shape[0]
    n_states = len(states)

    one_hot_encode = np.zeros((n_samples, n_states))

    for (i, sample) in enumerate(list(labels)):
        # Some tags are multiple targets, separated by a +
        tags_in_sample = sample.split('+')

        oh = np.zeros((1, n_states))
        for tag in tags_in_sample:
            if tag in states:
                oh[0, states.index(tag)] = 1

        # if target isn't linked to any state, put 1 in "unknown"
        if np.sum(oh) == 0:
            oh[0, -1] = 1

        one_hot_encode[i, :] = oh
    return one_hot_encode

def words_to_sentences(words):
    """
    @brief: takes a text as a list of words (or 1d ndarray) and return an list of sentences,
            as well as the index of the corresponding sentences for each word (in 1d array)
    @param:
        words: list (or 1d array) of string, the list of words we want to sentence-tokenize

    @return:
        list_sentences: list of list, text divided in sentences (each list being a sentence)
        indexes_word2sentences: 1d ndarray, index of the list associated to each word of words

    """
    n_words = len(words)
    text = ' '.join(words)
    list_sentences = list(sent_tokenize(text))

    current_idx_word = 0
    indexes_word2sentences = np.zeros((n_words,))
    for (i, sentence) in enumerate(list_sentences):
        # We split again our sentence
        words_in_sentence = sentence.split(' ')

        # for each word of the current sentence, match the correct word in words list and increment
        # the index of the words we're tagging
        for (j, w) in enumerate(words_in_sentence):
            # Sanity check, should always pass
            if w == words[current_idx_word]:
                indexes_word2sentences[current_idx_word] = i
                current_idx_word += 1
            else:
                print('error:', i, j, current_idx_word)

    return list_sentences, indexes_word2sentences



if __name__ == '__main__':

    # # For each db, encode the target into one_hot representation, according to states
    # STATES = ['howard_wolowitz', 'sheldon_cooper', 'leonard_hofstadter',
    #           'penny', 'rajesh_koothrappali']
    #
    # PATH = '/home/bsarthou/Documents/Ecole/AIC/TC3/db_projet/'
    # onlyfiles = [f for f in listdir(PATH) if isfile(join(PATH, f))]
    #
    # for file in onlyfiles:
    #     db = np.load(PATH+file)
    #     labels = db[:, 8]
    #     ohe = one_hot_encoding(labels, STATES)
    #     # print(ohe.shape, np.atleast_2d(labels).transpose().shape)
    #     print(np.concatenate((np.atleast_2d(labels).transpose(), ohe), axis=1))

    PATH = 'data/'
    onlyfiles = [f for f in listdir(PATH) if isfile(join(PATH, f))]

    for file in onlyfiles:
        db = np.load(PATH+file)
        sentences, index = words_to_sentences(db[:, 5])
