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

DATABASE_PATH = 'data/tbbt_db.csv'
DB_COLUMNS = ['season', 'episode', 'scene', 'scene_id', 'word', 'locutor']

PERSONS = ['Sheldon', 'Leonard', 'Penny', 'Raj', 'Howard']
UNKNOWN_STATE = ['Unknown']

def load_db():
    with open(DATABASE_PATH, "r", newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='§')
        return list(reader)


def load_sentences_by_person(states=PERSONS, filter=True):
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
    state_space = PERSONS + UNKNOWN_STATE
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
            person = UNKNOWN_STATE[0]

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
