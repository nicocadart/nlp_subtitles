import csv
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


SEASON = 0
EPISODE = 1
SCENE = 2
SCENE_ID = 3
PERSON = 4
SENTENCE = 5

DATABASE_PATH  = 'data/tbbt_db.csv'
DB_COLUMNS = ['season', 'episode', 'scene', 'scene_id', 'locutor', 'text']
DELIMITER = '§'

PERSONS = ['Sheldon', 'Leonard', 'Penny', 'Raj', 'Howard']
UNKNOWN_STATE = 'Unknown'


def load_db():
    """
    Load the full database.
    :return: list of list, where each element has format ['season', 'episode', 'scene', 'scene_id', 'locutor', 'text']
    """
    with open(DATABASE_PATH, "r", newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=DELIMITER)
        return list(reader)


def get_persons_scenes(db):
    """
    Parse database and return persons and text for each scene.
    :param db: the full database, returned by load_db()
    :return: persons: list of list, where each element is a person talking during the given scene
    :return: text: list of strings, where each element is the full text of the given scene
    """
    persons = []
    text = []
    scene_ids = []

    current_scene_id = 'x'
    current_scene_persons = set()
    current_scene_text = ""

    for db_entry in db:

        # if new scene
        if db_entry[SCENE_ID] != current_scene_id:
            if current_scene_id != "x":
                persons.append(list(current_scene_persons))
                text.append(current_scene_text)
                scene_ids.append(current_scene_id)
            current_scene_id = db_entry[SCENE_ID]
            current_scene_persons = set()
            current_scene_text = ""

        # if same scene
        current_scene_persons.add(db_entry[PERSON])
        current_scene_text += db_entry[SENTENCE]

    # don't forget to add last scene
    persons.append(list(current_scene_persons))
    text.append(current_scene_text)
    scene_ids.append(current_scene_id)

    return persons, text, scene_ids


def load_sentences_by_person(states=PERSONS, filter=False):
    """
    @brief: load a list of episodes files, and create a database of the sentences,
                associated with the labels being the person who said it
    @param:
            states: list of string, with the accepted names for the labels,
                    any others will be marked as UNKNOWN_STATE

    @return:
            sentences: list of string, each string being a complete sentence
            scenes_id: list of string, each string being the unique id of a scene in the corpus
            labels: list of int, each int being the integer associated
                                with class in PERSONS+(UNKNOWN_STATE)
    """
    sentences, scenes_id, labels = [], [], []
    state_space = PERSONS + [UNKNOWN_STATE]
    # Loading the list of sentences
    db = load_db()

    for sentence in db:
        if filter:
            filt_sentence = filter_sentence(sentence[SENTENCE])
        else:
            filt_sentence = sentence[SENTENCE]
        sentences.append(filt_sentence)
        scenes_id.append(sentence[SCENE_ID])

        # Check if the person is an accepted state
        person = sentence[PERSON]
        if person not in states:
            person = UNKNOWN_STATE

        label = state_space.index(person)
        labels.append(label)

    return sentences, scenes_id, labels


def filter_sentence(sentence, filter='nltk'):
    """
    @brief: take a sentence (string) as input, and return a string with the filtered sentence
    @param:
        sentence:

    @return:
        sentences: list of string, each string being a complete sentence
        scenes_id: list of string, each string being the unique id of a scene in the corpus
        labels: list of int, each int being the integer associated
                            with class in PERSONS+(UNKNOWN_STATE)
    """

    word_tokens = word_tokenize(sentence)
    if filter == 'nltk':
        # Create stop words list
        stop_words = set(stopwords.words('english'))
        filtered_list = []
        # Get trhough sentence and discard words in stop_words
        for w in word_tokens:
            if w not in stop_words:
                filtered_list.append(w)
    else:
        raise ValueError('WARNING: not implemented')

    filtered_sentence = ' '.join(filtered_list)

    return filtered_sentence
