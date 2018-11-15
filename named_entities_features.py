import numpy as np
from os.path import isfile
import csv

from stanfordcorenlp import StanfordCoreNLP
from sklearn.feature_extraction.text import CountVectorizer

from parsing_toolbox import load_db, get_persons_scenes, PERSONS, UNKNOWN_STATE

# CoreNLP params
CORENLP_RAM = '4g'
CORENLP_PATH = '/media/nicolas/Data/Documents/Scolarité/ENSTA/3A/AIC_UPSay/cours/TC3/stanford-corenlp-full-2018-02-27'

# path to csv files where pre-computed features are stored
PERSONS_NE_DB = "data/features_ne.csv"
PERSONS_NE_PUNCT_DB = "data/features_ne_punct.csv"
PERSONS_NE_INTERJ_DB = "data/features_ne_interj.csv"
PERSONS_NE_COREF_DB = "data/features_ne_coref.csv"

# categories of named entities to keep
NAMED_ENTITIES_CATEGORIES = ['PERSON', 'ORGANIZATION', 'NATIONALITY', 'LOCATION']

# number of samples to consider (between 0 and 2665, -1 is all)
N_SAMPLES = -1

# flags
USE_COREFS = True
REBUILD_FEATURES = False
PRINT_STATS_ON_FEATURES = True


def write_features_to_csv(csv_path, scene_ids, scene_locutors, scene_features, scene_features2=None):
    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='§')
        if scene_features2:
            for scene_id, locutors, features1, features2 in zip(scene_ids, scene_locutors, scene_features, scene_features2):
                features1_str = "|".join(features1)
                features2_str = "|".join(features2)
                locutors_str = "|".join(locutors)
                writer.writerow([scene_id, locutors_str, features1_str, features2_str])
        else:
            for scene_id, locutors, features in zip(scene_ids, scene_locutors, scene_features):
                features_str = "|".join(features)
                locutors_str = "|".join(locutors)
                writer.writerow([scene_id, locutors_str, features_str])


def read_features_from_csv(csv_path, has_2_features=False):
    scenes_ids, scenes_persons, scenes_features, scenes_features2 = [], [], [], []
    with open(csv_path, "r", newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='§')
        for row in reader:
            scene_id = row[0]
            locutors = row[1].split("|")
            features = row[2].split("|")
            scenes_ids.append(scene_id)
            scenes_persons.append(locutors)
            scenes_features.append(features)
            if len(row) > 3:
                has_2_features = True
            if has_2_features:
                features2 = row[3].split("|")
                scenes_features2.append(features2)
    if has_2_features:
        return scenes_ids, scenes_persons, [scenes_features, scenes_features2]
    else:
        return scenes_ids, scenes_persons, scenes_features


def filter_tags(tagged_tokens, categories=None, ignore_categories=('O',), replace_by=None):
    """ Filter list of tags ('token', 'tag') by selecting only special tags categories, or by ignoring others.
    If replace_by is not set, the returned list contains only filtered tuples. If it is defined, the rejected tags will
    be set to replace_by value."""
    if categories:
        categories = set(categories)
        if not replace_by:
            return [(token, tag) for token, tag in tagged_tokens if (tag in categories)]
        else:
            return [(token, tag) if (tag in categories) else (token, replace_by) for token, tag in tagged_tokens]
    else:
        ignore_categories = set(ignore_categories)
        if not replace_by:
            return [(token, tag) for token, tag in tagged_tokens if (tag not in ignore_categories)]
        else:
            return [(token, tag) if (tag not in ignore_categories) else (token, replace_by) for token, tag in tagged_tokens]


def filter_ne_punct(pos_tags, named_entities, null_ne='O', punct_tags=(',', '.')):
    """ Return list of named entities that are followed by punctuation mark"""
    ne_punct = []
    punct_tags = set(punct_tags)
    # loop on all but last index
    for i_token, token in enumerate(named_entities[:-1]):
        word, ne = token
        if ne != null_ne and pos_tags[i_token+1][1] in punct_tags:
            ne_punct.append(word)
    return ne_punct


def filter_ne_interj(pos_tags, named_entities, null_ne='O', neighborhood=(-2,2)):
    """ Return list of named entities that are in the neighborhood of an interjection"""
    ne_interj = []
    n_tokens = len(pos_tags)
    # loop over all named entities
    for i_token, token in enumerate(named_entities[:-1]):
        word, ne_tag = token
        # if current token is a named entity
        if ne_tag != null_ne:
            # extract pos tags neighborhood
            min_neighbor_id = max(0, i_token+neighborhood[0])
            max_neighbor_id = min(n_tokens-1, i_token+neighborhood[1])
            neighbors_pos = {pos for _, pos in pos_tags[min_neighbor_id:max_neighbor_id+1]}
            # if interjection in neighborhood
            if 'UH' in neighbors_pos:
                ne_interj.append(word)
    return ne_interj


def filter_ne_coref(corefs, named_entities,
                    positive_coref_set={'i', 'you', 'me', 'my', 'your', 'yourself'},
                    negative_coref_set={'he', 'she', 'his', 'her', 'him', 'himself', 'herself'}):
    """ Return 2 lists of named entities coreferenced to the positive or the negative sets."""
    positive_ne_coref, negative_ne_coref = [], []

    # keep only coreferences with detected named entities
    ne_corefs = []
    for coref in corefs:
        entities = {token[3] for token in coref}
        if not entities.isdisjoint(set(named_entities)):
            ne_corefs.append(coref)

    # loop on each coreference
    ne_lower = [word.lower() for word in named_entities]
    for coref in ne_corefs:
        references = {word.lower() for _, _, _, word in coref}

        new_pos_ne_coref, new_neg_ne_coref = set(), set()
        for i_name, name in enumerate(ne_lower):
            if name in references:
                if not references.isdisjoint(negative_coref_set):
                    new_neg_ne_coref.add(named_entities[i_name])
                if not references.isdisjoint(positive_coref_set):
                    new_pos_ne_coref.add(named_entities[i_name])

        positive_ne_coref += list(new_pos_ne_coref)
        negative_ne_coref += list(new_neg_ne_coref)

    return positive_ne_coref, negative_ne_coref


def build_ne_db(n_samples=N_SAMPLES):
    """
    @brief: build a list of episodes files, and create a database of the named entities in each scene,
                associated with the labels being the person who said it
    @return: named_entities : list of found named entities in each scene
             labels: list of locutors in each scene
    """

    # Load subtitles dataset
    db = load_db()
    scenes_persons, scenes_text, scene_ids = get_persons_scenes(db)
    n_samples = len(scenes_persons) if n_samples == -1 else n_samples

    # load CoreNLP
    nlp = StanfordCoreNLP(CORENLP_PATH, memory=CORENLP_RAM)

    # named entities features
    ne_pred_all = []
    ne_pred_punct = []
    ne_pred_interj = []
    if USE_COREFS:
        ne_pred_coref_pos = []
        ne_pred_coref_neg = []

    for i_scene, (persons, text) in enumerate(zip(scenes_persons[:n_samples], scenes_text[:n_samples])):

        # print("\n----------------------------")
        print("Building sample {}/{} ({}%)".format(i_scene+1, n_samples, round(100*(i_scene+1)/n_samples)), end='\r')
        # print(persons)
        # print(text)

        # run tagging on text
        pos_tags = nlp.pos_tag(text)
        named_entities = nlp.ner(text)

        # keep only named entities from several categories, and replace the others by 'O'
        ne_all_replaced = filter_tags(named_entities,
                                      categories=NAMED_ENTITIES_CATEGORIES,
                                      replace_by='O')
        ne_all = filter_tags(named_entities,
                             categories=NAMED_ENTITIES_CATEGORIES)
        ne_all = [word for word, _ in ne_all]
        # print("All named entities :\n{}".format(ne_all))

        # named entities followed by punctation
        ne_punct = filter_ne_punct(pos_tags, ne_all_replaced, null_ne='O', punct_tags=[',', '.'])
        # print("Named entities followed by punctuation :\n{}".format(ne_punct))

        # named entities preceded by interjection in a 2 neighborhood
        ne_interj = filter_ne_interj(pos_tags, ne_all_replaced, null_ne='O', neighborhood=(-2, 2))
        # print("Named entities near interjection :\n{}".format(ne_interj))

        # store predictions
        ne_pred_all.append(ne_all)
        ne_pred_punct.append(ne_punct)
        ne_pred_interj.append(ne_interj)

        # named entities linked to coreferences
        if USE_COREFS:
            if ne_all:
                corefs = nlp.coref(text)
                ne_pos_coref, ne_neg_coref = filter_ne_coref(corefs, ne_all)
            else:
                ne_pos_coref, ne_neg_coref = [], []
            ne_pred_coref_pos.append(ne_pos_coref)
            ne_pred_coref_neg.append(ne_neg_coref)

            # print('Named entities linked to coreferences :')
            # print(' positive : {}'.format(ne_pos_coref))
            # print(' negative : {}'.format(ne_neg_coref))

    # Do not forget to close! The backend server will consume a lot memory.
    nlp.close()

    # save detections to csv files for future executions
    print("\nSaving all named entities dataset to file \'{}\'".format(PERSONS_NE_DB))
    write_features_to_csv(PERSONS_NE_DB, scene_ids, scenes_persons, ne_pred_all)
    print("Saving named entities followed by punctuation dataset to file \'{}\'".format(PERSONS_NE_PUNCT_DB))
    write_features_to_csv(PERSONS_NE_PUNCT_DB, scene_ids, scenes_persons, ne_pred_punct)
    print("Saving named entities near interjection dataset to file \'{}\'".format(PERSONS_NE_INTERJ_DB))
    write_features_to_csv(PERSONS_NE_INTERJ_DB, scene_ids, scenes_persons, ne_pred_interj)
    if USE_COREFS:
        print("Saving named entities linked to coreferences to dataset to file \'{}\'".format(PERSONS_NE_COREF_DB))
        write_features_to_csv(PERSONS_NE_COREF_DB, scene_ids, scenes_persons, ne_pred_coref_pos, ne_pred_coref_neg)


def load_ne_features(filter='all', rebuild=False):
    """
    Load full named_entities and locutors per scene dataset, either by reading existing file or building it.
    :param filter: 'all', 'punct', 'interj'
    :return: named_entities : list of found named entities in each scene
             persons: list of locutors in each scene
             scene_ids: the alpha numeric string to identify each scene
    """
    # check database to read
    if filter == "all":
        db_file_path = PERSONS_NE_DB
    elif filter == "punct":
        db_file_path = PERSONS_NE_PUNCT_DB
    elif filter == "interj":
        db_file_path = PERSONS_NE_INTERJ_DB
    elif filter == "coref":
        db_file_path = PERSONS_NE_COREF_DB
    else:
        raise ValueError("Unknown 'filter' value. Must be in {'all', 'punct', 'interj'}")

    # build all datasets if not already exists
    if not isfile(db_file_path) or rebuild:
        print("Building named entities datasets...")
        build_ne_db()

    # Load specific dataset if exists
    scenes_ids, scenes_persons, scenes_ne = [], [], []
    if isfile(db_file_path):
        print("Loading named entities dataset from file \'{}\'".format(db_file_path))
        scenes_ids, scenes_persons, scenes_ne = read_features_from_csv(db_file_path)
        # filter locutors
        scenes_persons = [list({person if person in PERSONS else UNKNOWN_STATE for person in persons}) for persons in scenes_persons]

    return scenes_ids, scenes_persons, scenes_ne


def compute_detections_stats(real_persons, predicted_persons):
    """
    Compute stats on names predictions
    :param real_persons: list of list of strings; each string being the name of a person supposed to be in that scene
    :param predicted_persons: list of list of strings; each string being the name of a person supposed to be in that
                              scene. Must be of the same size as real_persons.
                              ex: [['Sheldon', 'Howard'], ['Raj']]
    :return: floats; accuracy, precision, recall
    """
    # predictions counts
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    # loop on all samples
    for i_scene, persons in enumerate(real_persons):
        # ground truth
        real_locutors = list(np.unique([person for person in persons if person in PERSONS]))
        # predictions
        pred_locutors = list(np.unique([person for person in predicted_persons[i_scene] if person in PERSONS]))
        # counts update
        tp += len([person for person in pred_locutors if person in real_locutors])
        fp += len([person for person in pred_locutors if person not in real_locutors])
        fn += len([person for person in real_locutors if person not in pred_locutors])
        tn += len([person for person in PERSONS if (person not in pred_locutors) and (person not in real_locutors)])
    # metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return accuracy, precision, recall


def get_ne_dataset(possible_locutors=PERSONS,
                   ne_all=True,
                   ne_punct=True,
                   ne_interj=True,
                   ne_coref=False,
                   return_scenes_ids=False,
                   return_vocab=False,
                   min_df=0.02,
                   binary=False):

    if not (ne_all or ne_punct or ne_interj or ne_coref):
        raise ValueError("At least one feature type is needed.")

    # build X : merge X_all, X_punct and X_interj
    X = {locutor: np.array([]) for locutor in possible_locutors}
    X_vocab = {locutor: [] for locutor in possible_locutors}

    # convert scenes text to bag-of-words
    vectorizer = CountVectorizer(min_df=min_df, stop_words='english', max_features=None)

    # build X for all named entities : bag of words
    if ne_all:
        scenes_ids, scenes_persons, scenes_ne_all = load_ne_features(filter="all")
        ne_all_strings = [" ".join(ne_list) for ne_list in scenes_ne_all]
        X_ne_all = vectorizer.fit_transform(ne_all_strings).toarray()
        X_ne_all_vocab = vectorizer.get_feature_names()
        for locutor in possible_locutors:
            X[locutor] = np.hstack([X[locutor], X_ne_all]) if X[locutor].size else X_ne_all
            X_vocab[locutor] = X_vocab[locutor] + X_ne_all_vocab

    # build X for named entities followed by punctation : bag of words
    if ne_punct:
        scenes_ids, scenes_persons, scenes_ne_punct = load_ne_features(filter="punct")
        ne_punct_strings = [" ".join(ne_list) for ne_list in scenes_ne_punct]
        X_ne_punct = vectorizer.fit_transform(ne_punct_strings).toarray()
        X_ne_punct_vocab = vectorizer.get_feature_names()
        for locutor in possible_locutors:
            X[locutor] = np.hstack([X[locutor], X_ne_punct]) if X[locutor].size else X_ne_punct
            X_vocab[locutor] = X_vocab[locutor] + X_ne_punct_vocab

    # build X for named entities near interjection : bag of words
    if ne_interj:
        scenes_ids, scenes_persons, scenes_ne_interj = load_ne_features(filter="interj")
        ne_interj_strings = [" ".join(ne_list) for ne_list in scenes_ne_interj]
        X_ne_interj = vectorizer.fit_transform(ne_interj_strings).toarray()
        X_ne_interj_vocab = vectorizer.get_feature_names()
        for locutor in possible_locutors:
            X[locutor] = np.hstack([X[locutor], X_ne_interj]) if X[locutor].size else X_ne_interj
            X_vocab[locutor] = X_vocab[locutor] + X_ne_interj_vocab

    # build X for named entities linked to present/absent person : one hot encoding
    if ne_coref:
        scenes_ids, scenes_persons, [scenes_ne_coref_pos, scenes_ne_coref_neg] = load_ne_features(filter="coref")
        for locutor in possible_locutors:
            ne_coref_pos = [1 if locutor in coref else 0 for coref in scenes_ne_coref_pos]
            ne_coref_neg = [1 if locutor in coref else 0 for coref in scenes_ne_coref_neg]
            # X_ne_coref = np.array([ne_coref_pos, ne_coref_neg]).T
            X_ne_coref = np.array([ne_coref_pos]).T
            X[locutor] = np.hstack([X[locutor], X_ne_coref]) if X[locutor].size else X_ne_coref
            X_vocab[locutor] = X_vocab[locutor] + [locutor, locutor]

    # convert to binary features
    if binary:
        for locutor in possible_locutors:
            X[locutor] = np.minimum(X[locutor], 1)

    # build y : one hot encoded persons
    y = {}
    for locutor in possible_locutors:
        y[locutor] = np.array([1 if locutor in persons else 0 for persons in scenes_persons])

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


# if called from command line, build features and print stats
if __name__ == "__main__" :

    # load or build named entities features
    scenes_ids, scenes_persons, ne_pred_all = load_ne_features(filter="all", rebuild=REBUILD_FEATURES)
    _, _, ne_pred_punct = load_ne_features(filter="punct")
    _, _, ne_pred_interj = load_ne_features(filter="interj")
    if USE_COREFS:
        _, scenes_persons, [ne_pred_coref_pos, ne_pred_coref_neg] = load_ne_features(filter="coref")

    # print predictions stats
    if PRINT_STATS_ON_FEATURES:
        print("\n============================\n")
        print("Detections of real locutors by named entities detection ({} samples used) :".format(len(scenes_persons)))

        print("\nAll named entities :")
        accuracy, precision, recall = compute_detections_stats(scenes_persons, ne_pred_all)
        print(" * precision = {:.3f}".format(precision))
        print(" * recall    = {:.3f}".format(recall))
        print(" * accuracy  = {:.3f}".format(accuracy))

        print("\nNamed entities followed by punctuation :")
        accuracy, precision, recall = compute_detections_stats(scenes_persons, ne_pred_punct)
        print(" * precision = {:.3f}".format(precision))
        print(" * recall    = {:.3f}".format(recall))
        print(" * accuracy  = {:.3f}".format(accuracy))

        print("\nNamed entities with interjection in neighborhood :")
        accuracy, precision, recall = compute_detections_stats(scenes_persons, ne_pred_interj)
        print(" * precision = {:.3f}".format(precision))
        print(" * recall    = {:.3f}".format(recall))
        print(" * accuracy  = {:.3f}".format(accuracy))

        print("\nNamed entities with positive coreferences :")
        accuracy, precision, recall = compute_detections_stats(scenes_persons, ne_pred_coref_pos)
        print(" * precision = {:.3f}".format(precision))
        print(" * recall    = {:.3f}".format(recall))
        print(" * accuracy  = {:.3f}".format(accuracy))

        print("\nNamed entities with negative coreferences :")
        accuracy, precision, recall = compute_detections_stats(scenes_persons, ne_pred_coref_neg)
        print(" * precision = {:.3f}".format(precision))
        print(" * recall    = {:.3f}".format(recall))
        print(" * accuracy  = {:.3f}".format(accuracy))
