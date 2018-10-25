import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from parsing_toolbox import load_sentences_persons
from keras.utils import to_categorical

################################
######## CONSTANTS

# id of episodes we use for learn, test
EPISODES_LEARN = [1, 2, 3, 4, 5, 6, 7, 8]
EPISODES_TEST = [9, 10, 11, 12]
# Name of the persons we want to detect (+ state 'unknown')
PERSONS = ['howard_wolowitz', 'sheldon_cooper', 'leonard_hofstadter', 'penny',
           'rajesh_koothrappali']
# Dir for embeddings data
GLOVE_DIR = 'data/'
# Size of embedding space
EMBEDDING_DIM = 300


################################
######## LOADING DATA FOR TRAIN

sentences, labels, _, _ = load_sentences_persons(EPISODES_LEARN, states=PERSONS)

maxlen = 500  # We will cut sentence after 500 words (max is 461))
max_words = 10000  # We will only consider the top 10,000 words in the dataset

# HERE texts are a list of sentences
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# Split the data into a training set and a validation set
# But first, shuffle the data, since we started from data
# where sample are ordered (all negative first, then all positive).
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
labels = to_categorical(labels)

n_classes = len(PERSONS) + 1
n_samples = len(data)

x_train = data[:round(0.8*n_samples)]
y_train = labels[:round(0.8*n_samples)]
x_val = data[round(0.8*n_samples):]
y_val = labels[round(0.8*n_samples):]

# Loading pre-embedding data
embeddings_index = {}
# WARNING watch the embedding dim
f = open(os.path.join(GLOVE_DIR, 'glove.6B.'+str(EMBEDDING_DIM)+'d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((max_words, EMBEDDING_DIM))

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < max_words:
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector


model = Sequential()
model.add(Embedding(max_words, EMBEDDING_DIM, input_length=maxlen))
model.add(Flatten())
# model.add(Dense(10000, activation='relu'))
# model.add(Dense(5000, activation='relu'))
# model.add(Dense(1000, activation='relu'))
# model.add(Dense(2048, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(n_classes, activation='sigmoid'))
model.summary()
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])
# history = model.fit(x_train, y_train,
#                     epochs=10,
#                     batch_size=32,
#                     validation_data=(x_val, y_val))
# model.save_weights('pre_trained_glove_model.h5')


# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(1, len(acc) + 1)
#
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
#
# plt.figure()
#
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
#
# plt.show()

sentences_test, labels_test, n_ep_test, n_scene_test = load_sentences_persons(EPISODES_TEST)

n_ep_test=np.array(n_ep_test).astype(float).astype(int)
n_scene_test=np.array(n_scene_test).astype(float).astype(int)

sequences_test = tokenizer.texts_to_sequences(sentences_test)
x_test = pad_sequences(sequences_test, maxlen=maxlen)
y_test = np.asarray(labels_test)
y_test = to_categorical(y_test)
print(x_test.shape, y_test.shape)

model.load_weights('pre_trained_glove_model.h5')
#print('TEST ACCURACY ON SENTENCE:', model.evaluate(x_test, y_test)[1])
threshold_prediction = 0.02

confusion_per_character = np.zeros((n_classes, 2, 2))
for ep in range(len(n_ep_test)):
    idx_ep = n_ep_test==ep
    ep_scene = n_scene_test[idx_ep]

    for sc in list(np.unique(ep_scene)):
        #print('({}, {})'.format(ep, sc))
        idx_scene = n_scene_test==sc
        id_scene = np.logical_and(idx_ep, idx_scene)
        # print(id_scene)
        x_scene = x_test[id_scene]
        y_scene = y_test[id_scene]

        predict_scene_by_sentence = np.array(model.predict(x_scene))
        predict_scene = np.sum(predict_scene_by_sentence, axis=0)/predict_scene_by_sentence.shape[0]
        truth_class = np.unique(np.argmax(y_scene, axis=1))
        predict_class = predict_scene[predict_scene>threshold_prediction].argsort()

        for character in range(n_classes):
            if character in truth_class and character in predict_class:
                confusion_per_character[character, 0, 0] += 1
            elif character in truth_class and character not in predict_class:
                confusion_per_character[character, 0, 1] += 1
            elif character not in truth_class and character in predict_class:
                confusion_per_character[character, 1, 0] += 1
            elif character not in truth_class and character not in predict_class:
                confusion_per_character[character, 1, 1] += 1

PERSONSS = PERSONS + ['unknown']
print('ACCURACY')
for character in range(n_classes):
    m_confusion = confusion_per_character[character, :, :]
    #print(m_confusion)
    print('{}: {:.4f}\n'.format(PERSONSS[character],
                              (m_confusion[0,0]+m_confusion[1,1])/m_confusion.sum()))
