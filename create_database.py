import csv
from os import listdir
from os.path import isfile, join, isdir

SUBTITLES_DIR = "TBBT/transcripts"
DATABASE_PATH  = 'data/tbbt_db.csv'

NEW_SCENE = "Scene"
DELIMITER = '§'

if __name__ == "__main__":

    db_full = []

    # loop over seasons
    seasons_dir = [f for f in listdir(SUBTITLES_DIR) if isdir(join(SUBTITLES_DIR, f))]
    for season_dir in seasons_dir:
        i_season = season_dir.split("Season")[-1]

        # loop over episodes
        episodes_files = [f for f in listdir("{}/{}".format(SUBTITLES_DIR, season_dir)) if
                          isfile("{}/{}/{}".format(SUBTITLES_DIR, season_dir, f))]
        for episodes_file in episodes_files:

            # get episode number
            if int(i_season) == 1:
                i_episode = episodes_file[-2:]
            else:
                i_episode = episodes_file.split("-")[3]

            with open("{}/{}/{}".format(SUBTITLES_DIR, season_dir, episodes_file)) as csvfile:
                reader = csv.reader(csvfile, delimiter=':', quotechar="|")
                i_scene = 0
                scene_id = "{}_{}_{}".format(i_season, i_episode, i_scene)
                for row in reader:

                    if len(row) == 0:
                        continue

                    # if new scene
                    elif row[0] == NEW_SCENE:
                        i_scene += 1

                    # if dialogue
                    else:
                        scene_id = "{}_{}_{}".format(i_season, i_episode, i_scene)
                        locutor = row[0]
                        sentences = "".join(row[1:])

                        db_full.append([i_season, i_episode, str(i_scene), scene_id, locutor, sentences])

    # write database
    with open(DATABASE_PATH, "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=DELIMITER)
        for row in db_full:
            writer.writerow(row)
