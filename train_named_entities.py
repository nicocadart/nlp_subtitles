import numpy as np
from scipy.optimize import minimize
import csv
import warnings

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss
from xgboost import XGBClassifier

from parsing_toolbox import PERSONS, UNKNOWN_STATE
from named_entities_toolbox import get_train_test_ne_persons_dataset


TEST_RESULTS_PATH = "data/prediction_ne_test.csv"


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
            # models[i_model] = (model_name, CalibratedClassifierCV(models[i_model][1], method='isotonic', cv=3))
            # model = models[i_model][1]
            model.fit(X_train, y_train)

            if verbose and X_valid.any() and y_valid.any():
                print(' * Accuracy : {:.2f}%'.format(100*model.score(X_valid, y_valid)))
                print(' * Logloss  : {:.3f}'.format(log_loss(y_valid, model.predict_proba(X_valid))))


def models_predict_proba(models, X):
    """ Run predictions for a list of models """
    y_proba_pred = []
    for model in models:
        y_proba_pred.append(model.predict_proba(X))
    return models


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
    n_classes = models[-1].n_classes_
    y_pred_p = np.zeros((X.shape[0], n_classes))
    for i_model, model in enumerate(models):
        y_pred_p += weights[i_model] * model.predict_proba(X)
    return y_pred_p


def train_model_mix(models, X, y, score='logloss'):
    """
    @brief: Load gridsearch result in .csv file
    @param:
            models: list of sklearn models (with params already passed) as a
                    list of tuple (name, model)

            X_train: ndarray (n_samples, n_features), array of samples to train
            y_train: ndarray (n_samples,), array of targets for each train sample

            X_val: ndarray (n_samples, n_features), array of samples to test
            y_val: ndarray (n_samples,), array of targets for each test sample

    @return:
            (print confusion matrix and log loss score for the final model (weighted prediction))
            models: list of learned sklearn models as a list of tuple (name, model)
            optimal_weights: weights for ponderation between model prediction,
                             optimized for those models
     """
    # predict proba
    y_proba_pred = models_predict_proba(models, X)

    def log_loss_func(weights):
        y_pred_mix = modelmix_predict_proba(models, weights, X)
        return log_loss(y, y_pred_mix)

    def error_rate_func(weights):
        y_pred_mix = modelmix_predict_proba(models, weights, X)
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

    min_count = 10
    once = False
    possible_locutors = PERSONS + [UNKNOWN_STATE]
    ne_dataset = get_train_test_ne_persons_dataset(possible_locutors, ne_min_count=min_count, once=once)
    X_train, y_train, ids_train, X_valid, y_valid, ids_valid, X_test, y_test, ids_test = ne_dataset

    print("Dimensions of datasets :")
    print(" * train : {}".format(X_train.shape))
    print(" * valid : {}".format(X_valid.shape))
    print(" * test  : {}".format(X_test.shape))

    # =================================================================
    #                      TRAIN CLASSIFIERS
    # =================================================================

    test_results = np.zeros((X_test.shape[0], 2*len(possible_locutors)))

    # Train models for each person and print results
    for i_person, person in enumerate(possible_locutors):
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

        clfs_names.append('LogisticRegression')
        parameters = {}
        clfs.append(LogisticRegression(**parameters))

        clfs_names.append('DecisionTreeClassifier 1')
        parameters = {'max_depth':None,
                      'min_samples_split':3}
        clfs.append(DecisionTreeClassifier(**parameters))

        clfs_names.append('RandomForestClassifier 1')
        parameters = {'n_estimators': 50}
        clfs.append(RandomForestClassifier(**parameters))

        clfs_names.append('RandomForestClassifier 2')
        parameters = {'n_estimators': 100}
        clfs.append(RandomForestClassifier(**parameters))

        clfs_names.append('MLPClassifier')
        parameters = {'hidden_layer_sizes': (50, 25, 5),
                      'activation':'logistic',
                      'early_stopping': True}
        clfs.append(MLPClassifier(**parameters))

        clfs_names.append('XGBoost')
        parameters = {'objective': 'binary:logistic',
                      'n_estimators': 150,
                      'max_depth': 9,
                      'learning_rate': 0.1,
                      'subsample': 0.7,
                      'colsample_bytree': 0.8,
                      'reg_lambda': 0,
                      'reg_alpha': 1,
                      'n_jobs': -1}
        clfs.append(XGBClassifier(**parameters))

        # -------------------
        #  Train classifiers
        # -------------------
        print("\nTraining classifiers independently")
        train_models(clfs, X_train, y_train[person], X_valid, y_valid[person], models_names=clfs_names)


        # --------------------------------
        #  Find ensamble learning weights
        # --------------------------------
        print("\nEnsemble learning")

        # get optimal weights
        best_weights = train_model_mix(clfs, X_valid, y_valid[person], score='logloss')
        y_pred_proba = modelmix_predict_proba(clfs, best_weights, X_valid)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # print results
        print(" * Best Weights   : {}".format(np.round(best_weights, 3)))
        print(" * Final accuracy : {:.2f}%".format(100*accuracy_score(y_valid[person], y_pred)))
        print(" * Final log-loss : {:.3f}".format(log_loss(y_valid[person], y_pred_proba)))
        print("{}".format(confusion_matrix(y_test[person], y_pred)))

        # save test results
        test_results[:, i_person] = y_test[person]
        test_results[:, len(possible_locutors)+i_person] = y_pred_proba[:, 1]

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
