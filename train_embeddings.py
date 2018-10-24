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


EPISODES_LEARN = [1, 2, 3, 4, 5, 6, 7, 8]
EPISODES_TEST = [9, 10, 11, 12]
GLOVE_DIR = 'data/'
EMBEDDING_DIM = 100

sentences, labels = load_sentences_persons(EPISODES_LEARN)

maxlen = 500  # We will cut sentence after 461 words (max is 461))
training_samples = 3000  # We will be training on 200 samples
validation_samples = len(sentences)-3001  # We will be validating on 10000 samples
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
n_classes = len(np.unique(labels))
labels = to_categorical(labels)

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]
# Loading pre-embedding data
embeddings_index = {}
# WARNING watch the embedding dim
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
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
model.add(Dense(32, activation='relu'))
model.add(Dense(n_classes, activation='sigmoid'))
model.summary()
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.h5')


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

sentences_test, labels_test = load_sentences_persons(EPISODES_LEARN)

sequences_test = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences_test, maxlen=maxlen)
y_test = np.asarray(labels)
y_test = to_categorical(y_test)

model.load_weights('pre_trained_glove_model.h5')
print(model.evaluate(x_test, y_test))
