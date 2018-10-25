import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from parsing_toolbox_new import load_sentences_by_person
from embeddings_tools import tokenize_corpus, compute_embedding_weights
from embeddings_tools import create_simple_model, create_conv_model


################################
######## CONSTANTS
# Characters we want to detect
PERSONS = ['Sheldon', 'Leonard', 'Penny', 'Raj', 'Howard']
# Dir for embeddings data
GLOVE_DIR = 'data/'
# Size of embedding space
EMBEDDING_DIM = 100
TRAIN_VALID_TEST_RATIO = (0.8, 0.1, 0.1)

MAXLEN = 250  # We will cut sentence after 250 words (max is 202))
MAX_WORDS = 10000  # We will only consider the top 10,000 words in the dataset

TRAIN = True # Launch a training on the data. If false, load latest trained model

################################
######## LOADING DATA FOR TRAIN

# those are list of the sentences, with associated id for the scene accross corpus,
# and the label (person speaking) associated
sentences, id_scene, labels = load_sentences_by_person(states=PERSONS, filter=False)

############################################
######## TOKENIZE CORPUS TO LEARN EMBEDDINGS

sequences, word_index = tokenize_corpus(sentences, num_words=MAX_WORDS)

data = pad_sequences(sequences, maxlen=MAXLEN)

labels = np.asarray(labels)
id_scene = np.asarray(id_scene)

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

############################################
######## TRAIN VALID TEST SPLIT

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
id_scene = id_scene[indices]

labels = to_categorical(labels)

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

print('TRAIN SHAPE:', x_train.shape, y_train.shape)
print('VAL SHAPE:', x_val.shape, y_val.shape)
print('TEST SHAPE:', x_test.shape, y_test.shape)

############################################
######## LOAD EMBEDDINGS

embedding_matrix = compute_embedding_weights(GLOVE_DIR, EMBEDDING_DIM, MAX_WORDS, word_index)

############################################
######## CREATE MODEL
# model = create_simple_model(MAX_WORDS, EMBEDDING_DIM, MAXLEN, embedding_matrix, n_classes)
model = create_conv_model(MAX_WORDS, EMBEDDING_DIM, MAXLEN, embedding_matrix, n_classes)

if TRAIN:
    # Train Model
    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=32,
                        validation_data=(x_val, y_val),
                        verbose=1)
    model.save_weights('pre_trained_glove_model_conv.h5')

    ## Plot train, validation accuracy and loss
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

############################################
######## TEST ACCURACY PER CHARACTER

model.load_weights('pre_trained_glove_model_conv.h5')

print('TEST ACCURACY ON SENTENCE:', model.evaluate(x_test, y_test)[1])
threshold_prediction = 0.02

confusion_per_character = np.zeros((n_classes, 2, 2))
unique_id_test = np.unique(id_test)

for id_sc in unique_id_test:
    # print('ID SCENE: {}'.format(id_sc))
    idx_scene = (id_test == id_sc)
    x_scene = x_test[idx_scene]
    y_scene = y_test[idx_scene]

    predict_scene_by_sentence = np.array(model.predict(x_scene))
    predict_scene = np.sum(predict_scene_by_sentence, axis=0)/predict_scene_by_sentence.shape[0]
    truth_class = np.unique(np.argmax(y_scene, axis=1))
    predict_class = predict_scene[predict_scene > threshold_prediction].argsort()

    # print(truth_class)
    # print(predict_class)

    for character in range(n_classes):
        if character in truth_class and character in predict_class:
            confusion_per_character[character, 0, 0] += 1
        elif character in truth_class and character not in predict_class:
            confusion_per_character[character, 0, 1] += 1
        elif character not in truth_class and character in predict_class:
            confusion_per_character[character, 1, 0] += 1
        elif character not in truth_class and character not in predict_class:
            confusion_per_character[character, 1, 1] += 1

STATES = PERSONS + ['Unknown']
print('ACCURACY')
for character in range(n_classes):
    m_confusion = confusion_per_character[character, :, :]
    print('{}: {:.4f}\n Confusion matrix: {}'.format(STATES[character],
                                                     (m_confusion[0, 0]+m_confusion[1, 1])/m_confusion.sum(),
                                                     m_confusion))
