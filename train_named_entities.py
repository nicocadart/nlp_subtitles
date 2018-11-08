import numpy as np
from collections import Counter
from scipy.optimize import minimize

import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss

from parsing_toolbox import *
from encoding import *


INDEX_SETS_PATH = "train_test_split_scenes_indices.npy"
PERSONS_NE_DB = "data/persons_ne.csv"
TEST_RESULTS_PATH = "data/prediction_named_entities_test.csv"


def get_named_entities(tokens, category=None, tokenize=False, once=True):
    """
    Return named entities found in some text.
    :param tokens: array or list of words. If a full text is given (without words being segmented, use tokenize=True)
    :param category: if provided, return only the list of named entities of type 'category'
    :param tokenize: needs to be set to True if 'tokens' is a raw text given as input.
    :param once: if True, each named entities is returned only once (remove several occurences)
    :return: if category=None: a dic with keys as detected named entities types, and values as list of detections.
             else: a list of detections of type 'category'
    """
    # tokenize raw text if necessary
    if tokenize:
        tokens = nltk.word_tokenize(tokens)

    # get named entities
    ne_chunk_tree = nltk.ne_chunk(nltk.pos_tag(tokens))
    ne_dict = {chunk.label(): [] for chunk in ne_chunk_tree if hasattr(chunk, 'label')}
    for chunk in ne_chunk_tree:
        if hasattr(chunk, 'label'):
            ne_dict[chunk.label()] += [c[0] for c in chunk]

    # filter named entities
    if once:
        for ne in ne_dict.keys():
            ne_dict[ne] = list(set(ne_dict[ne]))

    # return named entities
    if category:
        return ne_dict[category] if category in ne_dict.keys() else []
    else:
        return ne_dict


def build_ne_persons_dataset():
    """
    @brief: build a list of episodes files, and create a database of the named entities in each scene,
                associated with the labels being the person who said it
    @return: named_entities : list of found named entities in each scene
             labels: list of locutors in each scene
    """
    named_entities = []
    db = load_db()
    scenes_persons, scenes_text, scene_ids = get_persons_scenes(db)
    for persons, text in zip(scenes_persons, scenes_text):

        # get named entities
        ne_dict = get_named_entities(text, once=False, tokenize=True)
        ne = [word for words in ne_dict.values() for word in words]

        # update dataset
        named_entities.append(ne)

    return named_entities, scenes_persons, scene_ids


def load_ne_persons_dataset():
    """
    Load full named_entities and locutors per scene dataset, either by reading existing file or building it.
    :return: named_entities : list of found named entities in each scene
             persons: list of locutors in each scene
    """
    # read database
    named_entities, persons, scene_ids = [], [], []

    # Load dataset if already exists
    if isfile(PERSONS_NE_DB):
        print("Loading named entities dataset from pre-built file \'{}\'".format(PERSONS_NE_DB))
        with open(PERSONS_NE_DB, "r", newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='§')
            for row in reader:
                scene_id = row[0]
                locutors = row[1].split("|")
                ne = row[2].split("|")
                scene_ids.append(scene_id)
                named_entities.append(ne)
                persons.append(locutors)

    else:
        # build dataset
        print("Building named entities dataset")
        named_entities, persons, scene_ids = build_ne_persons_dataset()

        # write dataset for future executions
        print("Saving named entities dataset to file \'{}\'".format(PERSONS_NE_DB))
        with open("data/persons_ne.csv", "w", newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter='§')
            for scene_id, ne, locutors in zip(scene_ids, named_entities, persons):
                ne_str = "|".join(ne)
                locutors_str = "|".join(locutors)
                writer.writerow([scene_id, locutors_str, ne_str])

    return named_entities, persons, scene_ids


def clean_ne_persons_dataset(named_entities, persons, min_ne_count=5, states=PERSONS, unknown_state=UNKNOWN_STATE, once=False):
    ne_count = Counter([ne for ne_scene in named_entities for ne in ne_scene])
    ne_vocab = {ne for ne, count in ne_count.items() if count >= min_ne_count}

    ne_cleaned, persons_cleaned = [], []
    for i_scene in range(len(named_entities)):

        # clean labels
        locutors = list({person if person in states else unknown_state for person in persons[i_scene]})

        # clean named entities
        if once:
            ne = list({ne for ne in named_entities[i_scene] if ne in ne_vocab})
        else:
            ne = [ne for ne in named_entities[i_scene] if ne in ne_vocab]

        ne_cleaned.append(ne)
        persons_cleaned.append(locutors)

    return ne_cleaned, persons_cleaned


def get_train_test_ne_persons_dataset(named_entities, persons, scene_ids, train_val_test_path=INDEX_SETS_PATH, possible_locutors=PERSONS):
    """
    Convert ne/persons database into a trainable hot encoding database split into train, validation and test sets.
    :param named_entities:
    :param persons:
    :param train_ratio:
    :param possible_locutors:
    :return:
    """
    # constants
    n_samples = len(named_entities)
    scene_ids = np.array(scene_ids)
    persons = np.array(persons)
    indexes = np.load(train_val_test_path)
    train_ids, valid_ids, test_ids = indexes[0], indexes[1], indexes[2]

    # build X : hot encoded named_entities
    ne_vocab = {ne for ne_scene in named_entities for ne in ne_scene if ne}
    n_features = len(ne_vocab)
    X = np.zeros((n_samples, n_features))
    for i_scene in range(n_samples):
        X[i_scene, :] = np.sum(one_hot_encoding(named_entities[i_scene], list(ne_vocab), unknown_name=''), axis=0)
    X_train = X[np.isin(scene_ids, train_ids), :]
    X_valid = X[np.isin(scene_ids, valid_ids), :]
    X_test  = X[np.isin(scene_ids, test_ids),  :]

    for idx in test_ids:
        if idx not in scene_ids:
            print(idx)

    # build y : one hot encoded persons
    y_train, y_valid, y_test = {}, {}, {}
    for person in possible_locutors:
        y_train[person] = np.array([1 if person in locutors else 0 for locutors in persons[np.isin(scene_ids, train_ids)]])
        y_valid[person] = np.array([1 if person in locutors else 0 for locutors in persons[np.isin(scene_ids, valid_ids)]])
        y_test[person]  = np.array([1 if person in locutors else 0 for locutors in persons[np.isin(scene_ids, test_ids)]])

    return X_train, y_train, train_ids, X_valid, y_valid, valid_ids, X_test, y_test, test_ids


if __name__=="__main__":
    # params
    min_count = 5
    once = False

    # =================================================================
    #                         LOAD DATASET
    # =================================================================

    # Load dataset
    named_entities_full, persons_full, scene_ids = load_ne_persons_dataset()

    # for min_count in [30]:

    print("Cleaning dataset with min_ne_count = {}".format(min_count))

    # Clean dataset :
    #   - replace all occurences of unkown characters by UNKOWN_STATE
    #   - remove named_entities counted less than 'min_count' times
    named_entities, persons = clean_ne_persons_dataset(named_entities_full,
                                                       persons_full,
                                                       min_ne_count=min_count,
                                                       once=once)

    # # Display remaining dataset
    # for ne, pers in zip(named_entities_cleaned, persons_cleaned):
    #     print("{:<85}: {}".format(" ".join(pers), " ".join(ne)))

    # Get train and test datasets
    possible_locutors = PERSONS + [UNKNOWN_STATE]
    X_train, y_train, ids_train, X_valid, y_valid, ids_valid, X_test, y_test, ids_test = get_train_test_ne_persons_dataset(named_entities,
                                                                                                                           persons,
                                                                                                                           scene_ids,
                                                                                                                           possible_locutors=possible_locutors)

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
        print("\nTraining classifiers for '{}'".format(person))

        # -------------------
        #  Init classifiers
        # -------------------
        clfs_names = ['SVM linear', 'SVM poly', 'SVM rbf', 'LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 'RandomForestClassifier', 'MLPClassifier']
        clfs = []
        clfs.append(SVC(kernel='linear', probability=True))
        clfs.append(SVC(kernel='poly', probability=True))
        clfs.append(SVC(kernel='rbf', probability=True))
        clfs.append(LogisticRegression())
        clfs.append(DecisionTreeClassifier(max_depth=15))
        clfs.append(RandomForestClassifier(n_estimators=20, random_state=1337))
        clfs.append(RandomForestClassifier(n_estimators=10, random_state=4141))
        clfs.append(MLPClassifier(hidden_layer_sizes=(50, 25, 5), activation='logistic', max_iter=300))

        # -------------------
        #  Train classifiers
        # -------------------
        print(" * Training classifiers independently")
        y_proba_valid = []
        for i_clf, clf in enumerate(clfs):
            _ = clf.fit(X_train, y_train[person])
            y_pred = clf.predict(X_test)
            print("   > {:<22} : {:.4f}".format(clfs_names[i_clf], accuracy_score(y_test[person], y_pred)))
            #print(confusion_matrix(y_test[person], y_pred))

        # --------------------------------
        #  Find ensamble learning weights
        # --------------------------------
        print(" * Learning ensamble weights")

        # predict proba
        y_proba_pred = []
        for clf in clfs:
            y_proba_pred.append(clf.predict_proba(X_valid))

        # function to minimize
        def log_loss_func(weights):
            ''' scipy minimize will pass the weights as a numpy array '''
            final_prediction = 0
            for weight, prediction in zip(weights, y_proba_pred):
                final_prediction += weight * prediction
            return log_loss(y_valid[person], final_prediction)
        def error_rate_func(weights):
            ''' scipy minimize will pass the weights as a numpy array '''
            final_prediction = 0
            for weight, prediction in zip(weights, y_proba_pred):
                final_prediction += weight * prediction
            return 1 - accuracy_score(y_test[person], np.argmax(final_prediction, axis=1))

        # minimize weights
        init_weights = np.random.uniform(0.3, 0.7, (len(y_proba_pred),))
        init_weights /= np.sum(init_weights)
        # adding constraints  and a different solver as suggested by user 16universe
        # https://kaggle2.blob.core.windows.net/forum-message-attachments/75655/2393/otto%20model%20weights.pdf?sv=2012-02-12&se=2015-05-03T21%3A22%3A17Z&sr=b&sp=r&sig=rkeA7EJC%2BiQ%2FJ%2BcMpcA4lYQLFh6ubNqs2XAkGtFsAv0%3D
        constraint = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
        # our weights are bound between 0 and 1
        bounds = [(0, None)] * len(y_proba_pred)
        # get best weights
        res = minimize(log_loss_func, init_weights, method='SLSQP', bounds=bounds, constraints=constraint)
        EL_weights = res['x']
        # print results
        print("   > Best Weights     : {}".format(np.round(EL_weights, 3)))
        print("   > Initial log-loss : {}".format(log_loss_func(init_weights)))
        print("   > Final log-loss   : {}".format(res['fun']))

        # ---------------
        #  Print results
        # ---------------
        # predict proba
        y_pred_p = np.zeros((len(y_test[person]), 2))
        for i_clf, clf in enumerate(clfs):
            y_pred_p += EL_weights[i_clf] * clf.predict_proba(X_test)
        y_pred = np.argmax(y_pred_p, axis=1)
        print(" * Accuracy on test set with ensamble learning : {:.4f}".format(accuracy_score(y_test[person], y_pred)))
        print("{}".format(confusion_matrix(y_test[person], y_pred)))

        # save test results
        test_results[:, i_person] = y_test[person]
        test_results[:, len(possible_locutors)+i_person] = y_pred_p[:, 1]

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

