from os import listdir
from os.path import isfile, join
import numpy as np


def one_hot_encoding(labels, states, unknown_name='unknown'):
    """@brief: Return one_hot encoding of an array of labels for a given state space of strings
        If a label isn't in the state space, it is labelled "Unknown"

        @param
        labels: ndarray, of size (n_samples, )
        states: list of strings of len n_states, each string being a state of your system

        @return:
        one_hot_encode: ndarray, of size (n_samples, n_states + 1)
    """
    # Adding the "unknown" state to the state space
    states = (states + [unknown_name]) if unknown_name else states
    n_samples = len(labels)
    n_states = len(states)

    one_hot_encode = np.zeros((n_samples, n_states))

    for i, sample in enumerate(labels):

        if sample in states:
            one_hot_encode[i, states.index(sample)] = 1

        # if target isn't linked to any state, put 1 in "unknown"
        elif unknown_name:
            one_hot_encode[i, -1] = 1

    return one_hot_encode


if __name__ == '__main__':

    # For each db, encode the target into one_hot representation, according to states
    STATES = ['howard_wolowitz', 'sheldon_cooper', 'leonard_hofstadter',
              'penny', 'rajesh_koothrappali']

    PATH = 'data/'
    onlyfiles = [f for f in listdir(PATH) if isfile(join(PATH, f))]

    for file in onlyfiles:
        db = np.load(PATH+file)
        labels = db[:, 8]
        ohe = one_hot_encoding(labels, STATES)
        # print(ohe.shape, np.atleast_2d(labels).transpose().shape)
        print(np.concatenate((np.atleast_2d(labels).transpose(), ohe), axis=1))

    # PATH = 'data/'
    # onlyfiles = [f for f in listdir(PATH) if isfile(join(PATH, f))]
    #
    # for file in onlyfiles:
    #     db = np.load(PATH+file)
    #     sentences, index = words_to_sentences(db[:, 5])
