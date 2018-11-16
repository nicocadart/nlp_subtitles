# Détection des personnages présents dans les scènes d'une série

Projet réalisé dans le cadre du cours TC3 du Master 2 AIC à l'Université Paris-Sud.

Auteurs : Nicolas Cadart et Benoit Sarthou

Date : Novembre 2018

## Utilisation

Les transcripts des sous-titres de la série "The Big Bang Theory" sont nécessaires.

1. Créer un dossier `data` dans ce répertoire.
1. Générer le jeu de données : `python3 create_database.py`, puis `python3 create_train_test_scenes_split.py` 
2. Pré-calculer les features sur les entités nommées : `python3 named_entities_features.py`. Attention, l'exécution nécessite au moins 4GB de RAM et peut prendre au delà d'1h.
3. Lancer l'entraînement des modèles et la prédiction sur le jeu de test : `python3 train_models_and_predict_locutors.py`
