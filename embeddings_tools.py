import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.layers import Conv2D, Conv1D

def tokenize_corpus(sentences, num_words):

    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    return sequences, word_index

def compute_embedding_weights(directory, embedding_dim, max_words, word_index):

    embeddings_index = {}
    f = open(os.path.join(directory, 'glove.6B.'+str(embedding_dim)+'d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((max_words, embedding_dim))

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if i < max_words:
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

    return embedding_matrix


def create_simple_model(max_words, embedding_dim, maxlen, embedding_matrix, n_classes):
    """
    @brief: Create a Keras embedding model, and load the weights from embedding_matrix
    @param:
            max_words: int, max number of words in a sentence
            embedding_dim: int, dimension of embedding space
            maxlen: int, max length of sentence (width of input)
            embedding_matrix: ndarray (max_words, embedding_dim), precomputed weight for
                                the Embedded layer
            n_classes: int, number of class of the model

    @return:
        model: Keras model for the NN, compiled
    """

    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
    model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(n_classes, activation='sigmoid'))
    model.summary()

    # Fix weights of the embedding layer with the pre-trained ones
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False

    # Compile model
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    return model

def create_conv_model(max_words, embedding_dim, maxlen, embedding_matrix, n_classes):
    """
    @brief: Create a Keras embedding model, and load the weights from embedding_matrix
    @param:
            max_words: int, max number of words in a sentence
            embedding_dim: int, dimension of embedding space
            maxlen: int, max length of sentence (width of input)
            embedding_matrix: ndarray (max_words, embedding_dim), precomputed weight for
                                the Embedded layer
            n_classes: int, number of class of the model

    @return:
        model: Keras model for the NN, compiled
    """

    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
    # model.add(Flatten())
    model.add(Conv1D(128, 4))
    # model.add(Dense(128, activation='relu'))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(n_classes, activation='sigmoid'))
    model.summary()

    # Fix weights of the embedding layer with the pre-trained ones
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False

    # Compile model
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    return model
