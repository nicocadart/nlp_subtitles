import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import ExtraTreesClassifier

from parsing_toolbox import load_db, get_persons_scenes, PERSONS, UNKNOWN_STATE


def get_vocab_dataset(possible_locutors=PERSONS,
                      min_df=0.02,
                      max_features=50,
                      binary=False,
                      return_scenes_ids=False,
                      return_vocab=False):

    # load scenes and persons dataset
    scenes_persons, scenes_text, scenes_ids = get_persons_scenes(load_db())
    # filter locutors
    scenes_persons = [list({person if person in PERSONS else UNKNOWN_STATE for person in persons}) for persons in
                      scenes_persons]

    # build y : one hot encoded persons
    y = {}
    for locutor in possible_locutors:
        y[locutor] = np.array([1 if locutor in persons else 0 for persons in scenes_persons])

    # convert scenes text to bag-of-words
    vectorizer = CountVectorizer(min_df=min_df, stop_words='english', max_features=None)

    # build X : bag-of-words of best words for each locutor
    X, X_vocab = {}, {}
    for locutor in possible_locutors:
        # get words with DF > min_df and convert them to bag of words
        X[locutor] = vectorizer.fit_transform(scenes_text).toarray()
        X_vocab[locutor] = vectorizer.get_feature_names()

        # keep only most relevant words for this locutor
        clf = ExtraTreesClassifier(n_estimators=max_features)
        _ = clf.fit(X[locutor], y[locutor])
        best_features_ids = np.argsort(clf.feature_importances_)[::-1][:max_features]
        X[locutor] = X[locutor][:, best_features_ids]
        X_vocab[locutor] = [X_vocab[locutor][i_word] for i_word in best_features_ids]

    # convert to binary features
    if binary:
        for locutor in possible_locutors:
            X[locutor] = np.minimum(X[locutor], 1)

    if return_scenes_ids:
        if return_vocab:
            return X, y, scenes_ids, X_vocab
        else:
            return X, y, scenes_ids
    else:
        if return_vocab:
            return X, y, X_vocab
        else:
            return X, y
