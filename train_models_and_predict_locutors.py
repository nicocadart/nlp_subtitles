import numpy as np
from scipy.optimize import minimize
import csv
import warnings

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss, recall_score, precision_score
from xgboost import XGBClassifier

from parsing_toolbox import PERSONS, UNKNOWN_STATE, split_train_valid_test
from named_entities_features import get_ne_dataset
from vocabulary_features import get_vocab_dataset


POSSIBLE_LOCUTORS = PERSONS #+ [UNKNOWN_STATE]
TEST_RESULTS_PATH = "data/prediction_ne_test.csv"

USE_VOCAB = True
USE_NE_ALL = False
USE_NE_PUNCT = True
USE_NE_INTERJ = True
USE_NE_COREF = False
MIN_DF = 0.02
MAX_VOCAB_SIZE = 40
BINARY_FEATURES = False


def train_models(models, X_train, y_train, X_valid=None, y_valid=None, models_names=None, verbose=True):
    """ Train a list of models on training set."""
    for i_model, model in enumerate(models):

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)

            if models_names is not None:
                model_name = models_names[i_model]
            else:
                model_name = i_model + 1

            if verbose:
                print('Training model {}'.format(model_name))
            models[i_model] = CalibratedClassifierCV(model, method='isotonic', cv=3)
            model = models[i_model]
            model.fit(X_train, y_train)

            if verbose and X_valid.any() and y_valid.any():
                y_pred_p = model.predict_proba(X_valid)
                y_pred = model.predict(X_valid)
                print(' * Precision : {:.2f}%'.format(100 * precision_score(y_valid, y_pred)))
                print(' * Recall    : {:.2f}%'.format(100 * recall_score(y_valid, y_pred)))
                print(' * Accuracy  : {:.2f}%'.format(100 * accuracy_score(y_valid, y_pred)))
                # print(' * Logloss   : {:.3f}'.format(log_loss(y_valid, y_pred_p)))


def models_predict_proba(models, X):
    """ Run predictions for a list of models """
    y_proba_pred = []
    for model in models:
        y_proba_pred.append(model.predict_proba(X))
    return y_proba_pred


def models_predict_proba_stacked(models, X):
    y_pred_p_models = models_predict_proba(models, X)
    return np.hstack(tuple(y_pred_p[:,:-1] for y_pred_p in y_pred_p_models))


def modelmix_predict_proba(models, weights, X):
    """
    @brief: take a list of sklearn models, weights and a dataset and return the weighted prediction
            over the samples

    @param:
            models: list of tuple (name, model), with model a sklearn model already trained
            weights: list of float, weight for each model (sum(weight)==1)
            X: ndarray, (n_samples, n_features), dataset to predict

    @return:
            y_pred_p: ndarray, (n_samples, n_classes), probability for each class for each sample
    """
    n_classes = len(models[0].classes_)
    y_pred_p = np.zeros((X.shape[0], n_classes))
    for i_model, model in enumerate(models):
        y_pred_p += weights[i_model] * model.predict_proba(X)
    return y_pred_p


def train_model_mix(models, X, y, score='logloss'):
    """ Compute best weights for ensemble learning """
    n_classes = len(models[0].classes_)

    # predict proba
    y_proba_pred = models_predict_proba(models, X)

    def log_loss_func(weights):
        y_pred_mix = np.zeros((X.shape[0], n_classes))
        for i_model, model in enumerate(models):
            y_pred_mix += weights[i_model] * y_proba_pred[i_model]
        # y_pred_mix = modelmix_predict_proba(models, weights, X)
        return log_loss(y, y_pred_mix)

    def error_rate_func(weights):
        y_pred_mix = np.zeros((X.shape[0], n_classes))
        for i_model, model in enumerate(models):
            y_pred_mix += weights[i_model] * y_proba_pred[i_model]
        # y_pred_mix = modelmix_predict_proba(models, weights, X)
        return 1 - accuracy_score(y, np.argmax(y_pred_mix, axis=1))

    # function we want to minimize
    if score == 'logloss':
        opt_function = log_loss_func
    elif score == 'accuracy':
        opt_function = error_rate_func

    # Uniform initialisation
    init_weights = np.ones((len(y_proba_pred),)) / len(y_proba_pred)
    # Weights are in range [0; 1] and must sum to 1
    constraint = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    bounds = [(0, 1)] * len(y_proba_pred)
    # Compute best weights (method chosen with the advice of Kaggle kernel)
    res = minimize(opt_function, init_weights, method='SLSQP', bounds=bounds, constraints=constraint)
    optimal_weights = res['x']

    return optimal_weights


if __name__=="__main__":

    # =================================================================
    #                         LOAD DATASET
    # =================================================================

    print("\nLoading and preparing dataset for classification...")

    # init X
    X = {locutor: np.array([]) for locutor in POSSIBLE_LOCUTORS}
    X_map = {locutor: [] for locutor in POSSIBLE_LOCUTORS}

    # features based on whole vocabulary and best discriminating words
    if USE_VOCAB:
        X_vocab, y, scenes_ids, X_vocab_map = get_vocab_dataset(possible_locutors=POSSIBLE_LOCUTORS,
                                                                min_df=MIN_DF,
                                                                max_features=MAX_VOCAB_SIZE,
                                                                binary=BINARY_FEATURES,
                                                                return_scenes_ids=True,
                                                                return_vocab=True)
        for locutor in POSSIBLE_LOCUTORS:
            X[locutor] = np.hstack([X[locutor], X_vocab[locutor]]) if X[locutor].size else X_vocab[locutor]
            X_map[locutor] = X_map[locutor] + X_vocab_map[locutor]

    # features based on named entities
    if USE_NE_ALL or USE_NE_PUNCT or USE_NE_INTERJ or USE_NE_COREF:
        X_ne, y, scenes_ids, X_ne_map = get_ne_dataset(possible_locutors=POSSIBLE_LOCUTORS,
                                                       ne_all=USE_NE_ALL,
                                                       ne_punct=USE_NE_PUNCT,
                                                       ne_interj=USE_NE_INTERJ,
                                                       ne_coref=USE_NE_COREF,
                                                       min_df=MIN_DF,
                                                       binary=BINARY_FEATURES,
                                                       return_scenes_ids=True,
                                                       return_vocab=True)
        for locutor in POSSIBLE_LOCUTORS:
            X[locutor] = np.hstack([X[locutor], X_ne[locutor]]) if X[locutor].size else X_ne[locutor]
            X_map[locutor] = X_map[locutor] + X_ne_map[locutor]

    # separate train, valid and test sets using fixed ids
    X_train, X_valid, X_test = split_train_valid_test(X)
    y_train, y_valid, y_test = split_train_valid_test(y)
    ids_train, ids_valid, ids_test = split_train_valid_test(np.array(scenes_ids))

    # display dimensions of datasets
    print("\nDimensions of datasets :")
    print(" * train : {}".format(X_train[PERSONS[0]].shape))
    print(" * valid : {}".format(X_valid[PERSONS[0]].shape))
    print(" * test  : {}".format(X_test[PERSONS[0]].shape))

    # =================================================================
    #                      TRAIN CLASSIFIERS
    # =================================================================

    test_results = np.zeros((X_test[POSSIBLE_LOCUTORS[0]].shape[0], 2*len(POSSIBLE_LOCUTORS)))

    # Train models for each person and print results
    for i_person, person in enumerate(POSSIBLE_LOCUTORS):
        print("\n==================================================")
        print("{:^50}".format(person.upper()))
        print("==================================================")

        # -------------------
        #  Init classifiers
        # -------------------

        clfs_names = []
        clfs = []

        # clfs_names.append('SVM linear')
        # clfs.append(SVC(kernel='linear', C=1 ,probability=True))
        #
        # clfs_names.append('SVM poly')
        # clfs.append(SVC(kernel='poly', C=1, probability=True))
        #
        # clfs_names.append('SVM rbf')
        # clfs.append(SVC(kernel='rbf', C=1, probability=True))

        # clfs_names.append('LogisticRegression')
        # parameters = {'penalty': 'l2',
        #               'C': 0.1}
        # clfs.append(LogisticRegression(**parameters))

        # clfs_names.append('RandomForestClassifier 1')
        # parameters = {'min_samples_split': 2,
        #               'n_estimators': 200,
        #               'max_depth': 10,
        #               'criterion': 'entropy',
        #               'n_jobs': -1}
        # clfs.append(RandomForestClassifier(**parameters))

        clfs_names.append('RandomForestClassifier 2')
        parameters = {'min_samples_split': 4,
                      'n_estimators': 100,
                      'max_depth': 20,
                      'criterion': 'gini',
                      'n_jobs': -1}
        clfs.append(RandomForestClassifier(**parameters))

        # clfs_names.append('MLPClassifier 1')
        # parameters = {'activation': 'relu',
        #               'early_stopping': True,
        #               'hidden_layer_sizes': (90,),
        #               'batch_size': 128,
        #               'alpha': 0.0001}
        # clfs.append(MLPClassifier(**parameters))
        #
        # clfs_names.append('MLPClassifier 2')
        # parameters = {'activation': 'relu',
        #               'early_stopping': True,
        #               'hidden_layer_sizes': (70, 50, 25, 15),
        #               'batch_size': 128,
        #               'alpha': 0.001}
        # clfs.append(MLPClassifier(**parameters))
        #
        # clfs_names.append('SVM')
        # parameters = {'probability': True,
        #               'kernel': 'rbf',
        #               'C': 10.0}
        # clfs.append(SVC(**parameters))

        clfs_names.append('XGBoost 1')
        parameters = {'n_estimators': 100,
                      'objective': 'binary:logistic',
                      'colsample_bytree': 0.8,
                      'subsample': 1,
                      'max_depth': 9,
                      'reg_lambda': 1,
                      'n_jobs': -1,
                      'learning_rate': 0.1,
                      'reg_alpha': 0}
        clfs.append(XGBClassifier(**parameters))

        # clfs_names.append('XGBoost 2')
        # parameters = {'n_estimators': 200,
        #               'objective': 'binary:logistic',
        #               'colsample_bytree': 0.8,
        #               'subsample': 1,
        #               'max_depth': 9,
        #               'reg_lambda': 1,
        #               'n_jobs': -1,
        #               'learning_rate': 0.1,
        #               'reg_alpha': 1}
        # clfs.append(XGBClassifier(**parameters))

        # -------------------
        #  Train classifiers
        # -------------------
        print("\nTraining classifiers independently")
        train_models(clfs, X_train[person], y_train[person], X_valid[person], y_valid[person], models_names=clfs_names)


        # --------------------------------
        #  Find ensamble learning weights
        # --------------------------------
        # print("\nEnsemble learning")
        #
        # # get optimal weights
        # best_weights = train_model_mix(clfs, X_valid[person], y_valid[person], score='logloss')
        # y_pred_proba = modelmix_predict_proba(clfs, best_weights, X_test[person])
        # y_pred = np.argmax(y_pred_proba, axis=1)
        #
        # # print results
        # print(" * Best Weights  : {}".format(np.round(best_weights, 3)))
        # print(" * Test accuracy : {:.2f}%".format(100*accuracy_score(y_test[person], y_pred)))
        # print(" * Test log-loss : {:.3f}".format(log_loss(y_test[person], y_pred_proba)))
        # print("{}".format(confusion_matrix(y_test[person], y_pred)))

        # save test results
        test_results[:, i_person] = y_test[person]
        test_results[:, len(POSSIBLE_LOCUTORS)+i_person] = clfs[1].predict_proba(X_test[person])[:, 1]

        # -------------------------
        #  Second layer classifier
        # -------------------------
        # print("\nSecond layer of classifier")
        #
        # # concatenate predictions from 1st layer classifiers
        # X_train2 = models_predict_proba_stacked(clfs, X_train[person])
        # X_valid2 = models_predict_proba_stacked(clfs, X_valid[person])
        # X_test2 = models_predict_proba_stacked(clfs, X_test[person])

        # classifier
        # parameters = {'objective': 'binary:logistic',
        #               'n_estimators': 200,
        #               'max_depth': 9,
        #               'learning_rate': 0.1,
        #               'subsample': 0.7,
        #               'colsample_bytree': 0.8,
        #               'reg_lambda': 1,
        #               'reg_alpha': 0,
        #               'n_jobs': -1}
        # final_model = XGBClassifier(**parameters)

        # parameters = {'n_estimators': 200,
        #               'max_depth': 9,
        #               'min_samples_split': 2}
        # final_model = RandomForestClassifier(**parameters)
        #
        # # train on dataset and test predictions
        # final_model.fit(X_train2, y_train[person])
        # y_pred_proba = final_model.predict_proba(X_test2)
        # y_pred = np.argmax(y_pred_proba, axis=1)
        #
        # # print results
        # print(" * Test accuracy : {:.2f}%".format(100*accuracy_score(y_test[person], y_pred)))
        # print(" * Test log-loss : {:.3f}".format(log_loss(y_test[person], y_pred_proba)))
        # print("{}".format(confusion_matrix(y_test[person], y_pred)))


    # save results in csv
    with open(TEST_RESULTS_PATH, "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        for i_scene in range(len(ids_test)):
            writer.writerow([ids_test[i_scene]] + list(test_results[i_scene, :]))
    print("\nTests results saved to '{}'".format(TEST_RESULTS_PATH))

    # training_episodes_numbers = range(1,10)
    #
    # ne_all = []
    # text_scene = []
    #
    # for i_episode in training_episodes_numbers:
    #      episode = load_episode_db(i_episode)
    #      scenes_numbers = get_scenes_numbers(episode)
    #
    #      for scene in scenes_numbers:
    #          tokens = get_scene(episode, scene)
    #          real_locutors = np.unique(tokens[:, PERSON])
    #          ne = get_named_entities(tokens[:, WORD])
    #          ne_all += [word for words in ne.values() for word in words]
    #          text_scene.append(get_scene_text(tokens, scene))
    #          # print("SCENE {} :".format(int(scene)))
    #          # print(" * real locutors            : {}".format(" ".join(real_locutors)))
    #          # print(" * 'persons' named entities : {}\n".format(" ".join(ne['PERSON'])))
    #          # print(" * 'location' named entities : {}\n".format(" ".join(ne['LOC'])))
    #          # print(ne, '\n')
    #
    # # get stats
    # ne_counter = Counter(ne_all)
    # ne_set = set(ne_all)
    #
    # # words tfidf
    # tfidf_vec = TfidfVectorizer(stop_words='english')
    # tfidf = tfidf_vec.fit_transform(text_scene)
    # words_idf = dict(zip(tfidf_vec.get_feature_names(), tfidf_vec.idf_))
    #
    # # display results
    # print("number of named entities {} :".format(len(ne_set)))
    # for ne, count in ne_counter.most_common():
    #     print("  * {} : {}".format(ne, count))
