import numpy as np
from collections import Counter

import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC

from parsing_toolbox import *
from encoding import *


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


def load_ne_persons_dataset():
    """
    @brief: load a list of episodes files, and create a database of the named entities in each scene,
                associated with the labels being the person who said it
    @return: named_entities : list of found named entities in each scene
             labels: list of locutors in each scene
    """
    labels = []
    named_entities = []

    for i_episode in range(1, 10):
        episode = load_episode_db(i_episode)
        scenes_numbers = get_scenes_numbers(episode)

        for i_scene in scenes_numbers:
            tokens = get_scene(episode, i_scene)

            # get locutors
            locutors = list(np.unique(tokens[:, PERSON]))

            # get named entities
            ne_dict = get_named_entities(tokens[:, WORD], once=False)
            ne = [word for words in ne_dict.values() for word in words]

            # update dataset
            labels.append(locutors)
            named_entities.append(ne)

    return named_entities, labels


def clean_ne_persons_dataset(named_entities, persons, min_ne_count=1, states=PERSONS, unknown_state="unknown"):
    ne_count = Counter([ne for ne_scene in named_entities for ne in ne_scene])
    ne_vocab = {ne for ne, count in ne_count.items() if count >= min_ne_count}

    ne_cleaned, persons_cleaned = [], []
    for i_scene in range(len(named_entities)):

        # clean labels
        locutors = list({person if person in states else unknown_state for locutors in persons[i_scene] for person in locutors.split("+")})

        # clean named entities
        ne = list({ne for ne in named_entities[i_scene] if ne in ne_vocab})

        ne_cleaned.append(ne)
        persons_cleaned.append(locutors)

    return ne_cleaned, persons_cleaned


def learn_and_test_classification(named_entities, persons, train_index=80):
    # named_entities one hot encoding
    ne_vocab = {ne for ne_scene in named_entities_cleaned for ne in ne_scene}
    ne_oh = []
    for i_scene in range(len(named_entities_cleaned)):
        ne_oh.append(np.sum(one_hot_encoding(np.array(named_entities_cleaned[i_scene]), list(ne_vocab)), axis=0))

    # separate train and test datasets
    X = ne_oh
    X_train = X[:train_index]
    X_test = X[train_index:]

    # sheldon classifier
    y = [1 if 'sheldon_cooper' in locutors else 0 for locutors in persons_cleaned]
    y_train = y[:train_index]
    y_test = y[train_index:]
    sheldon_clf = SVC(kernel='linear')
    _ = sheldon_clf.fit(X_train, y_train)
    print(" * Sheldon classifier score : {}".format(sheldon_clf.score(X_test, y_test)))

    # leonard classifier
    y = [1 if 'leonard_hofstadter' in locutors else 0 for locutors in persons_cleaned]
    y_train = y[:train_index]
    y_test = y[train_index:]
    leonard_clf = SVC(kernel='linear')
    _ = leonard_clf.fit(X_train, y_train)
    print(" * Leonard classifier score : {}".format(leonard_clf.score(X_test, y_test)))

    # howard classifier
    y = [1 if 'howard_wolowitz' in locutors else 0 for locutors in persons_cleaned]
    y_train = y[:train_index]
    y_test = y[train_index:]
    howard_clf = SVC(kernel='linear')
    _ = howard_clf.fit(X_train, y_train)
    print(" * Howard classifier score  : {}".format(howard_clf.score(X_test, y_test)))

    # rajesh classifier
    y = [1 if 'rajesh_koothrappali' in locutors else 0 for locutors in persons_cleaned]
    y_train = y[:train_index]
    y_test = y[train_index:]
    raj_clf = SVC(kernel='linear')
    _ = raj_clf.fit(X_train, y_train)
    print(" * Rajesh classifier score  : {}".format(raj_clf.score(X_test, y_test)))

    # penny classifier
    y = [1 if 'penny' in locutors else 0 for locutors in persons_cleaned]
    y_train = y[:train_index]
    y_test = y[train_index:]
    penny_clf = SVC(kernel='linear')
    _ = penny_clf.fit(X_train, y_train)
    print(" * Penny classifier score   : {}".format(penny_clf.score(X_test, y_test)))



if __name__=="__main__":

    print("Building named entities dataset...")

    # get dataset
    named_entities, persons = load_ne_persons_dataset()

    for min_count in [1, 2, 3, 4, 5]:
        print("\nCleaning dataset with min_ne_count = {}".format(min_count))

        # clean dataset
        named_entities_cleaned, persons_cleaned = clean_ne_persons_dataset(named_entities, persons, min_ne_count=min_count)

        # for ne, pers in zip(named_entities_cleaned, persons_cleaned):
        #     print("{:<85}: {}".format(" ".join(pers), " ".join(ne)))

        print("Training models on dataset...")
        learn_and_test_classification(named_entities_cleaned, persons_cleaned, train_index=75)

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

