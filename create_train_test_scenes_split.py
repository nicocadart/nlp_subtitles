import numpy as np
from parsing_toolbox_new import load_db, SCENE_ID

RANDOM_SEED = 42
TRAIN_VALID_TEST_RATIO = (0.7, 0.15, 0.15)
OUTPUT_FILE_PATH = "train_test_split_scenes_indices.npy"


if __name__ == "__main__":

    # check args\n",
    assert sum(TRAIN_VALID_TEST_RATIO) == 1, "Sum of ratio must be equal to 1"
    assert min(TRAIN_VALID_TEST_RATIO) >= 0, "Each ratio value must be >= 0"

    # get scene ids
    db = load_db()
    scene_ids = np.unique([sample[SCENE_ID] for sample in db])

    # shuffle scene ids with fixed seed
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(scene_ids)

    # separate sets
    n_samples = len(scene_ids)
    train_index = round(TRAIN_VALID_TEST_RATIO[0] * n_samples)
    valid_index = train_index + round(TRAIN_VALID_TEST_RATIO[1] * n_samples)
    train_scenes_ids = scene_ids[:train_index]
    valid_scenes_ids = scene_ids[train_index:valid_index]
    test_scenes_ids  = scene_ids[valid_index:]

    # save datasets
    np.save("train_test_split_scenes_indices.npy", [train_scenes_ids, valid_scenes_ids, test_scenes_ids])