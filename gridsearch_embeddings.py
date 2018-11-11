import numpy as np
import csv
import matplotlib.pyplot as plt
import warnings

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from parsing_toolbox import load_sentences_by_person, PERSONS, UNKNOWN_STATE
from embeddings_toolbox import tokenize_corpus, compute_embedding_weights, return_embeddings


################################
######## CONSTANTS
# Characters we want to detect
STATES = PERSONS + [UNKNOWN_STATE]

# Dir for embeddings data
GLOVE_DIR = 'data/'
EMBEDD_PATH = 'data/embedding_matrix.npy'

INDEX_SETS_PATH = "data/train_test_split_scenes_indices.npy"
OUTPUT_PREDICTIONS_PATH = 'data/prediction_embeddings_test.csv'

# Size of embedding space
EMBEDDING_DIM = 100
TRAIN_VALID_TEST_RATIO = (0.8, 0.1, 0.1)
RANDOM_SPLIT = True #WARNING False value doesnt work for now on potato computer
MAXLEN = 200  # We will cut sentence after 200 words (max is 202))
MAX_WORDS = 10000  # We will only consider the top 10,000 words in the dataset

N_CROSS_VAL = 3
N_JOBS = -1
CSV_DIR = 'gridsearch_results'


################################
######## LOADING DATA FOR TRAIN

# those are list of the sentences, with associated id for the scene accross corpus,
# and the label (person speaking) associated
sentences, id_scene, labels = load_sentences_by_person(states=PERSONS)

############################################
######## TOKENIZE CORPUS TO LEARN EMBEDDINGS

sequences, word_index = tokenize_corpus(sentences, num_words=MAX_WORDS)

data = pad_sequences(sequences, maxlen=MAXLEN)


# WARNING Bad RAM again
data = data[:1000, :]
labels = labels[:1000]

###################################################
######## LOAD PRE-TRAINED EMBEDDINGS AND EMBED DATA

# From Glove pre-trained embedding
embedding_matrix = compute_embedding_weights(GLOVE_DIR, EMBEDDING_DIM, MAX_WORDS, word_index)

# # From personal train on our classification model
# embedding_matrix = np.load(EMBEDD_PATH)

X = return_embeddings(data, MAX_WORDS, EMBEDDING_DIM, MAXLEN, embedding_matrix)

labels = np.asarray(labels)
id_scene = np.asarray(id_scene)
# labels = to_categorical(labels)

print('Shape of data tensor:', X.shape)
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

    t_v_t_scene_indices = np.load(INDEX_SETS_PATH)
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


models = []
#
# # Logistic Regression
# parameters = {'penalty':['l1', 'l2'],
#               'C':[0.1, 1., 10.],
#               'multi_class':['ovr']}
# models.append(("LogisticRegression", LogisticRegression(), parameters))
#
# RandomForestClassifier
parameters = {'n_estimators':[10, 50, 100, 150, 200, 250, 300],
              'criterion':['gini', 'entropy'],
              'max_depth':[None, 5, 10, 20],
              'min_samples_split':[2, 4],
              'n_jobs': [1]}
models.append(("RandomForest", RandomForestClassifier(), parameters))
#
#
# # DecisionTree
# parameters = {'splitter':['best', 'random'],
#               'criterion':['gini', 'entropy'],
#               'max_depth':[None, 10, 20, 50],
#               'min_samples_split':[2, 4, 6]}
# models.append(("DecisionTree", DecisionTreeClassifier(), parameters))

# # KNeighbors
# parameters = {'n_neighbors':[3, 5, 7],
#               'p':[1, 2],
#               'n_jobs': [1]}
# models.append(("KNeighbors", KNeighborsClassifier(), parameters))

# Extreme Gradient Boosting classifier
# parameters = {'objective':['binary:logistic'],
#               'subsample':[0.7, 1],
#               'colsample_bytree':[0.8],
#               'learning_rate':[0.1],
#               'max_depth':[9],
#               'reg_alpha':[0,1],
#               'reg_lambda':[0,1],
#               'n_estimators':[100, 150],
#               'n_jobs': [-1]}
# models.append(("XGBoost", XGBClassifier(), parameters))
#
# # Multi-Layer Perceptron Classifier
# parameters = {'hidden_layer_sizes':[(90,), (50,), (40, 30), (50, 25, 15), (70, 50, 25, 15)],
# 'activation':['logistic', 'relu'],
# 'batch_size':[128, 256],
# 'alpha':[0.001, 0.0001],
# 'early_stopping':[True]}
# models.append(("MLPClassifier", MLPClassifier(), parameters))
# # SVM : /!\ Very slow convergence
# # parameters = {'kernel':['linear', 'rbf', 'poly', 'sigmoid'],
# #               'C':[0.1, 1., 10., 100],
# #               'probability':[True],
# #               'gamma'=['scale']}
# # models.append(("SVC", SVC(), parameters))

results = []
names = []
for name, model, parameter in models:
    print("\n==================================================")
    print("{:^50}".format(name))
    print("==================================================\n")

    # run grid search
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        clf = GridSearchCV(model, parameter, scoring='accuracy', cv=N_CROSS_VAL,
                           n_jobs=N_JOBS, verbose=2)
        clf.fit(X, labels)
        res = clf.cv_results_

    # save results to csv file
    with open('{}/{}.csv'.format(CSV_DIR, name), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        for nb_set_params in range(len(res['params'])):
            res_str = '{} {} {}'.format(res['params'][nb_set_params],
                                        res['mean_test_score'][nb_set_params],
                                        res['std_test_score'][nb_set_params])
            writer.writerow([res_str])
