import os
import csv
import numpy as np
import matplotlib.pyplot as plt
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


def train_model(model, x_train, y_train, x_val, y_val, epochs=4,
                plot=True, savepath='pre_trained_glove_model.h5'):

    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_data=(x_val, y_val),
                        verbose=1)
    if savepath is not None:
        model.save_weights(savepath)

    if plot:
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
        return 0


def test_model(model, x_test, y_test, id_test, n_classes, states,
               threshold_prediction = 0.02,
               loadpath='pre_trained_glove_model.h5',
               savepath='data/prediction_embeddings_test.csv'):

    model.load_weights(loadpath)
    confusion_per_character = np.zeros((n_classes, 2, 2))
    unique_id_test = np.unique(id_test)


    with open(savepath, "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['id_scene'] + states + states)

        for id_sc in unique_id_test:
            # print('ID SCENE: {}'.format(id_sc))
            idx_scene = (id_test == id_sc)
            x_scene = x_test[idx_scene]
            y_scene = y_test[idx_scene]

            predict_scene_by_sentence = np.array(model.predict(x_scene))
            predict_scene = np.sum(predict_scene_by_sentence, axis=0)/predict_scene_by_sentence.shape[0]

            truth_class = np.unique(np.argmax(y_scene, axis=1))
            truth_array = np.zeros((n_classes,))
            truth_array[truth_class] = 1

            row = [id_sc] + list(truth_array) + list(predict_scene)
            writer.writerow(row)

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

    print('ACCURACY')
    for character in range(n_classes):
        m_confusion = confusion_per_character[character, :, :]
        print('{}: {:.4f}\n Confusion matrix: {}'.format(states[character],
                                                         (m_confusion[0, 0]+\
                                                          m_confusion[1, 1])/m_confusion.sum(),
                                                         m_confusion))
    return confusion_per_character
