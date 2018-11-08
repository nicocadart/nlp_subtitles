import numpy as np
from scipy.optimize import minimize

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss

from parsing_toolbox import PERSONS, UNKNOWN_STATE
from named_entities_toolbox import get_train_test_ne_persons_dataset


TEST_RESULTS_PATH = "data/prediction_named_entities_test.csv"


if __name__=="__main__":

    # =================================================================
    #                         LOAD DATASET
    # =================================================================

    min_count = 5
    once = False
    possible_locutors = PERSONS + [UNKNOWN_STATE]
    X_train, y_train, ids_train, X_valid, y_valid, ids_valid, X_test, y_test, ids_test = get_train_test_ne_persons_dataset(possible_locutors, ne_min_count=10, once=False)

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

