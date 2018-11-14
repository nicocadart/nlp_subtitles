import numpy as np
import csv
import warnings

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from parsing_toolbox import PERSONS, UNKNOWN_STATE
from named_entities_toolbox import get_train_test_ne_persons_dataset
from train_vocabulary import get_train_test_vocabulary_dataset

POSSIBLE_LOCUTORS = PERSONS + [UNKNOWN_STATE]
DATASET = 'vocabulary'  # 'vocabulary' or 'named_entities'

# params for vocabulary dataset
MIN_DF = 0.02

# params for named entities dataset
NE_MIN_COUNT = 25
NE_ONCE = False

# params for gridsearch
N_CROSS_VAL = 3
N_JOBS = 7
CSV_DIR = 'gridsearch_results_vocab'


if __name__ == "__main__":

    # =================================================================
    #                         LOAD DATASET
    # =================================================================

    # load dataset
    if DATASET == 'vocabulary':
        vocab_dataset = get_train_test_vocabulary_dataset(POSSIBLE_LOCUTORS, min_df=MIN_DF)
        X_train, y_train, ids_train, X_valid, y_valid, ids_valid, X_test, y_test, ids_test = vocab_dataset

    elif DATASET == 'named_entities':
        ne_dataset = get_train_test_ne_persons_dataset(POSSIBLE_LOCUTORS, ne_min_count=NE_MIN_COUNT, once=NE_ONCE)
        X_train, y_train, ids_train, X_valid, y_valid, ids_valid, X_test, y_test, ids_test = ne_dataset

    print("Dimensions of datasets :")
    print(" * train : {}".format(X_train.shape))
    print(" * valid : {}".format(X_valid.shape))
    print(" * test  : {}".format(X_test.shape))

    # =================================================================
    #                      DEFINE CLASSIFIERS
    # =================================================================

    models = []

    # Logistic Regression
    parameters = {'penalty':['l1', 'l2'],
                  'C':[0.1, 1., 10.]}
                  # 'multi_class':['auto']}
    models.append(("LogisticRegression", LogisticRegression(), parameters))

    # RandomForestClassifier
    parameters = {'n_estimators':[50, 100, 150, 200],
                  'criterion':['gini', 'entropy'],
                  'max_depth':[None, 10, 20],
                  'min_samples_split':[2, 4],
                  'n_jobs': [1]}
    models.append(("RandomForest", RandomForestClassifier(), parameters))

    # Multi-Layer Perceptron Classifier
    parameters = {'hidden_layer_sizes':[(90,), (50,), (40, 30), (50, 25, 15), (70, 50, 25, 15)],
                  'activation':['logistic', 'relu'],
                  'batch_size':[128, 256],
                  'alpha':[0.001, 0.0001],
                  'early_stopping':[True]}
    models.append(("MLPClassifier", MLPClassifier(), parameters))

    # DecisionTree
    parameters = {'splitter':['best', 'random'],
                  'criterion':['gini', 'entropy'],
                  'max_depth':[None, 10, 20, 50],
                  'min_samples_split':[2, 4, 6]}
    models.append(("DecisionTree", DecisionTreeClassifier(), parameters))

    # KNeighbors
    parameters = {'n_neighbors':[3, 5, 7],
                  'p':[1, 2],
                  'n_jobs': [1]}
    models.append(("KNeighbors", KNeighborsClassifier(), parameters))

    # Extreme Gradient Boosting classifier
    parameters = {'objective':['binary:logistic'],
                  'subsample':[0.7, 1],
                  'colsample_bytree':[0.8],
                  'learning_rate':[0.1],
                  'max_depth':[9],
                  'reg_alpha':[0,1],
                  'reg_lambda':[0,1],
                  'n_estimators':[100, 150, 200],
                  'n_jobs': [1]}
    models.append(("XGBoost", XGBClassifier(), parameters))

    # SVM
    parameters = {'kernel':['linear', 'rbf', 'poly', 'sigmoid'],
                  'C':[0.1, 1., 10., 100.],
                  'probability': [True]}
    models.append(("SVC", SVC(), parameters))


    # =================================================================
    #                      TRAIN CLASSIFIERS
    # =================================================================

    # Train models for each person and print results
    for i_person, person in enumerate(POSSIBLE_LOCUTORS):
        print("\n==================================================")
        print("{:^50}".format(person.upper()))
        print("==================================================")

        # merge training and validation datasets for cross validation
        X = np.vstack((X_train, X_valid))
        y = np.concatenate((y_train[person], y_valid[person]))

        results = []
        names = []
        for name, model, parameter in models:
            print("\n----------------------------------------------------")
            print("{:^50}".format(name))
            print("--------------------------------------------------\n")

            # run grid search
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",category=FutureWarning)
                warnings.filterwarnings("ignore",category=DeprecationWarning)
                clf = GridSearchCV(model, parameter,
                                   scoring=['neg_log_loss', 'accuracy'],
                                   cv=N_CROSS_VAL,
                                   n_jobs=N_JOBS,
                                   verbose=2,
                                   refit=False,
                                   return_train_score=False)
                clf.fit(X, y)
                res = clf.cv_results_

            # save results to csv file
            with open('{}/{}.csv'.format(CSV_DIR, name), 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                for nb_set_params in range(len(res['params'])):
                    # res_str = '{} {} {}'.format(res['params'][nb_set_params],
                    #                             res['mean_test_score'][nb_set_params],
                    #                             res['std_test_score'][nb_set_params])
                    writer.writerow([person,
                                     res['params'][nb_set_params],
                                     res['mean_test_accuracy'][nb_set_params],
                                     res['std_test_accuracy'][nb_set_params],
                                     res['mean_test_neg_log_loss'][nb_set_params],
                                     res['std_test_neg_log_loss'][nb_set_params]])
