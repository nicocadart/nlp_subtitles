import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from parsing_toolbox_new import load_sentences_by_person, PERSONS, UNKNOWN_STATE
from embeddings_tools import tokenize_corpus, compute_embedding_weights, train_model, test_model
from embeddings_tools import create_simple_model, create_conv_model


################################
######## CONSTANTS
# Characters we want to detect
STATES = PERSONS + UNKNOWN_STATE
# Dir for embeddings data
GLOVE_DIR = 'data/'

# Size of embedding space
EMBEDDING_DIM = 100
TRAIN_VALID_TEST_RATIO = (0.8, 0.1, 0.1)
RANDOM_SPLIT = False
MAXLEN = 250  # We will cut sentence after 250 words (max is 202))
MAX_WORDS = 10000  # We will only consider the top 10,000 words in the dataset

TRAIN = False # Launch a training on the data. If false, load latest trained model

################################
######## LOADING DATA FOR TRAIN

# those are list of the sentences, with associated id for the scene accross corpus,
# and the label (person speaking) associated
sentences, id_scene, labels = load_sentences_by_person(states=PERSONS)

############################################
######## TOKENIZE CORPUS TO LEARN EMBEDDINGS

sequences, word_index = tokenize_corpus(sentences, num_words=MAX_WORDS)

data = pad_sequences(sequences, maxlen=MAXLEN)

labels = np.asarray(labels)
id_scene = np.asarray(id_scene)
labels = to_categorical(labels)

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

############################################
######## TRAIN VALID TEST SPLIT
if RANDOM_SPLIT:

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    id_scene = id_scene[indices]

    n_classes = len(PERSONS) + 1
    n_samples = len(data)

    int_train = round(TRAIN_VALID_TEST_RATIO[0]*n_samples)
    int_val = int_train + round(TRAIN_VALID_TEST_RATIO[1]*n_samples)

    x_train = data[:int_train]
    y_train = labels[:int_train]

    x_val = data[int_train:int_val]
    y_val = labels[int_train:int_val]

    x_test = data[int_val:]
    y_test = labels[int_val:]
    id_test = id_scene[int_val:]

else:
    n_classes = len(PERSONS) + 1

    t_v_t_scene_indices = np.load('train_test_split_scenes_indices.npy')
    train, val, test = t_v_t_scene_indices[0], t_v_t_scene_indices[1], t_v_t_scene_indices[2]

    x_train = data[np.isin(id_scene, train)]
    y_train = labels[np.isin(id_scene, train)]

    x_val = data[np.isin(id_scene, val)]
    y_val = labels[np.isin(id_scene, val)]

    x_test = data[np.isin(id_scene, test)]
    y_test = labels[np.isin(id_scene, test)]
    id_test = id_scene[np.isin(id_scene, test)]



print('TRAIN SHAPE:', x_train.shape, y_train.shape)
print('VAL SHAPE:', x_val.shape, y_val.shape)
print('TEST SHAPE:', x_test.shape, y_test.shape)

############################################
######## LOAD EMBEDDINGS

embedding_matrix = compute_embedding_weights(GLOVE_DIR, EMBEDDING_DIM, MAX_WORDS, word_index)

############################################
######## CREATE MODEL
model = create_simple_model(MAX_WORDS, EMBEDDING_DIM, MAXLEN, embedding_matrix, n_classes)
# model = create_conv_model(MAX_WORDS, EMBEDDING_DIM, MAXLEN, embedding_matrix, n_classes)

if TRAIN:
    # Train Model
    train_model(model, x_train, y_train, x_val, y_val, epochs=10)

############################################
######## TEST ACCURACY PER CHARACTER

test_model(model, x_test, y_test, id_test, n_classes, states=STATES,
           threshold_prediction=0.02,
           loadpath='pre_trained_glove_model.h5',
           savepath='data/prediction_embeddings_test.csv')
