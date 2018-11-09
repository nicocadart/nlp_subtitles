import numpy as np
from scipy.optimize import minimize
import csv
import warnings

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from parsing_toolbox import load_db, get_persons_scenes, PERSONS, UNKNOWN_STATE


MIN_DF = 0.02
POSSIBLE_LOCUTORS = PERSONS + [UNKNOWN_STATE]
INDEX_SETS_PATH = "data/train_test_split_scenes_indices.npy"
TEST_RESULTS_PATH = "data/prediction_vocabulary_test.csv"


def build_vocab_dataset(locutors=POSSIBLE_LOCUTORS, min_df=MIN_DF, return_mapping=False):

    # load scenes and persons dataset
    db = load_db()
    scenes_persons, scenes_text, scene_ids = get_persons_scenes(db)

    # convert scenes text to bag-of-words
    vectorizer = CountVectorizer(min_df=min_df, stop_words='english', max_features=None)
    X = vectorizer.fit_transform(scenes_text).toarray()
    X_map = vectorizer.get_feature_names()

    # convert persons to one-hot-encoding representation
    scenes_persons_cleaned = [ np.unique([person if person in PERSONS else UNKNOWN_STATE for person in persons]).tolist() for persons in scenes_persons]
    y = MultiLabelBinarizer(classes=locutors).fit_transform(scenes_persons_cleaned)
    y_map = locutors

    if return_mapping:
        return X, y, scene_ids, X_map, y_map
    else:
        return X, y, scene_ids


def get_train_test_vocabulary_dataset(possible_locutors, min_df=MIN_DF, return_mapping=False, train_val_test_path=INDEX_SETS_PATH):
    # load dataset
    if return_mapping:
        X, y, scene_ids, X_map, y_map = build_vocab_dataset(min_df=min_df, return_mapping=return_mapping)
    else:
        X, y, scene_ids = build_vocab_dataset(min_df=min_df, return_mapping=return_mapping)

    # split dataset
    indexes = np.load(train_val_test_path)
    train_ids, valid_ids, test_ids = indexes[0], indexes[1], indexes[2]

    # build X
    X_train = X[np.isin(scene_ids, train_ids), :]
    X_valid = X[np.isin(scene_ids, valid_ids), :]
    X_test = X[np.isin(scene_ids, test_ids), :]

    # reduce number of features
    # features_reducer = TruncatedSVD(n_components=200)
    features_reducer = PCA(n_components=250)
    X_train = features_reducer.fit_transform(X_train)
    X_valid = features_reducer.transform(X_valid)
    X_test = features_reducer.transform(X_test)

    # build y : one hot encoded persons
    y_train, y_valid, y_test = {}, {}, {}
    for i_person, person in enumerate(possible_locutors):
        y_train[person] = y[np.isin(scene_ids, train_ids), i_person]
        y_valid[person] = y[np.isin(scene_ids, valid_ids), i_person]
        y_test[person]  = y[np.isin(scene_ids, test_ids), i_person]

    if return_mapping:
        return X_train, y_train, train_ids, X_valid, y_valid, valid_ids, X_test, y_test, test_ids, X_map, y_map
    else:
        return X_train, y_train, train_ids, X_valid, y_valid, valid_ids, X_test, y_test, test_ids


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
                print(' * Accuracy : {:.2f}%'.format(100*model.score(X_valid, y_valid)))
                print(' * Logloss  : {:.3f}'.format(log_loss(y_valid, model.predict_proba(X_valid))))


def models_predict_proba(models, X):
    """ Run predictions for a list of models """
    y_proba_pred = []
    for model in models:
        y_proba_pred.append(model.predict_proba(X))
    return y_proba_pred


def models_predict_proba_satcked(models, X):
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

    vocab_dataset = get_train_test_vocabulary_dataset(POSSIBLE_LOCUTORS, min_df=MIN_DF, return_mapping=False, train_val_test_path=INDEX_SETS_PATH)
    X_train, y_train, ids_train, X_valid, y_valid, ids_valid, X_test, y_test, ids_test = vocab_dataset

    print("Dimensions of datasets :")
    print(" * train : {}".format(X_train.shape))
    print(" * valid : {}".format(X_valid.shape))
    print(" * test  : {}".format(X_test.shape))

    # =================================================================
    #                      TRAIN CLASSIFIERS
    # =================================================================

    test_results = np.zeros((X_test.shape[0], 2*len(POSSIBLE_LOCUTORS)))

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

        clfs_names.append('LogisticRegression')
        parameters = {'penalty': 'l2',
                      'C': 0.1}
        clfs.append(LogisticRegression(**parameters))

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
        train_models(clfs, X_train, y_train[person], X_valid, y_valid[person], models_names=clfs_names)


        # --------------------------------
        #  Find ensamble learning weights
        # --------------------------------
        print("\nEnsemble learning")

        # get optimal weights
        best_weights = train_model_mix(clfs, X_valid, y_valid[person], score='logloss')
        y_pred_proba = modelmix_predict_proba(clfs, best_weights, X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # print results
        print(" * Best Weights  : {}".format(np.round(best_weights, 3)))
        print(" * Test accuracy : {:.2f}%".format(100*accuracy_score(y_test[person], y_pred)))
        print(" * Test log-loss : {:.3f}".format(log_loss(y_test[person], y_pred_proba)))
        print("{}".format(confusion_matrix(y_test[person], y_pred)))

        # save test results
        test_results[:, i_person] = y_test[person]
        test_results[:, len(POSSIBLE_LOCUTORS)+i_person] = y_pred_proba[:, 1]

        # -------------------------
        #  Second layer classifier
        # -------------------------
        print("\nSecond layer of classifier")

        # concatenate predictions from 1st layer classifiers
        X_train2 = models_predict_proba_satcked(clfs, X_train)
        X_valid2 = models_predict_proba_satcked(clfs, X_valid)
        X_test2 = models_predict_proba_satcked(clfs, X_test)

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

        parameters = {'n_estimators': 200,
                      'max_depth': 9,
                      'min_samples_split': 2}
        final_model = RandomForestClassifier(**parameters)

        # train on dataset and test predictions
        final_model.fit(X_train2, y_train[person])
        y_pred_proba = final_model.predict_proba(X_test2)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # print results
        print(" * Test accuracy : {:.2f}%".format(100*accuracy_score(y_test[person], y_pred)))
        print(" * Test log-loss : {:.3f}".format(log_loss(y_test[person], y_pred_proba)))
        print("{}".format(confusion_matrix(y_test[person], y_pred)))


    # save results in csv
    with open(TEST_RESULTS_PATH, "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        for i_scene in range(len(ids_test)):
            writer.writerow([ids_test[i_scene]] + list(test_results[i_scene, :]))
    print("\nTests results saved to '{}'".format(TEST_RESULTS_PATH))
