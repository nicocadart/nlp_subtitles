import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.layers import Conv2D, Conv1D

WEIGHTS_PATH = 'data/trained_weights.h5'
EMBEDD_PATH = 'data/embedding_matrix.npy'
OUTPUT_PREDICTIONS_PATH = 'data/prediction_embeddings_test.csv'


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
                                the Embedded layer (no pre-trained weights if None)
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
    if embedding_matrix is not None:
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
    if embedding_matrix is not None:
        model.layers[0].set_weights([embedding_matrix])
        model.layers[0].trainable = False

    # Compile model
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    return model


def return_embeddings(data, max_words, embedding_dim, maxlen, embedding_matrix):

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
        embedded_data: ndarray (n_samples, embedding_dim), data expressed through embeddings
    """

    model_emb = Sequential()
    model_emb.add(Embedding(max_words, embedding_dim, input_length=maxlen))

    model_emb.layers[0].set_weights([embedding_matrix])

    model_emb.compile('rmsprop', 'mse')
    embedded_data = model_emb.predict(data)
    embedded_data = embedded_data.reshape(len(data), maxlen*embedding_dim)

    return embedded_data


def train_model(model, x_train, y_train, x_val, y_val, epochs=4,
                plot=True, model_path=WEIGHTS_PATH, embedding_path=EMBEDD_PATH):
    """
    @brief: Train a Keras model on train data, with validation, and plot the learning and
                logloss curve
    @param:
        model: Keras model for embeddings
        x_train: ndarray (n_train_samples, n_features),
                  data to train on (each sample is a sentence, encoded)
        y_train: ndarray (n__train_samples, n_classes), targets for train in categorical form
        x_val: ndarray (n_val_samples, n_features),
                  data to validate on
        y_val: ndarray (n__val_samples, n_classes), targets for validation in categorical form
        epochs: int, number of epochs for training
        plot: boolean, if True, plot the learning curve (train and val), as well as logloss
        model_path: str, path where the model is saved after training (no save if None)
        embedding_path: str, path where the embedding weights are saved
                        after training (no save if None)

    @return:
        model: Keras model for the NN, and trained
    """

    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_data=(x_val, y_val),
                        verbose=1)

    # Saving the model for later
    if model_path is not None:
        model.save_weights(model_path)

    # Saving the embedding layer for reuse
    if embedding_path is not None:
        embedding_matrix = model.layers[0].get_weights()[0]
        np.save(embedding_path, embedding_matrix)

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
        return model


def test_model(model, x_test, y_test, id_test, n_classes, states,
               threshold_prediction = 0.02,
               loadpath=WEIGHTS_PATH,
               savepath=OUTPUT_PREDICTIONS_PATH):

    """
    @brief: Test a Keras model on test data, and compute accuracy and confusion matrix per character
            for a scene
    @param:
        model: Keras model for embeddings
        x_test: ndarray (n_test_samples, n_features),
                  data to test on (each sample is a sentence, encoded)
        y_test: ndarray (n_test_samples, n_classes), targets for test in categorical form
        id_test: ndarray (n_test_samples, ), id of scene for each sentence of the test set
        n_classes: int, number of classes for prediction
        states: list of str, names of classes for prediction (including 'UNKNOWN_STATE')
        threshold_prediction: float, threshold of probability for classification
        loadpath: str, if not None, load weigths for Keras model
        savepath: str, path to save the results of prediction in .csv form

    @return:
        confusion_per_character: list of ndarray of size (2,2), list of confusion matrix
                                    for each state
    """

    if loadpath is not None:
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

            # print(predict_scene)
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
        print('{}: {:.4f}\n Confusion matrix: {}\ Precision: {}, Recall: {}'.format(states[character],
                                                                                    (m_confusion[0, 0]+\
                                                                                     m_confusion[1, 1])/m_confusion.sum(),
                                                                                    m_confusion,
                                                                                    m_confusion[0, 0]/(m_confusion[0, 0]+m_confusion[0, 1]),
                                                                                    m_confusion[0, 0]/(m_confusion[0, 0]+m_confusion[1, 0])))

    return confusion_per_character
