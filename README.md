# nlp_subtitles

## Presence of locutors in the corpus
```
Sheldon: 60.64%
Raj: 35.87%
Howard: 40.60%
Unknown: 61.95%
Leonard: 54.82%
Penny: 42.55%
```

## Features selection

### Named entities

#### Results without learning

```
Detections of real locutors by named entities detection (2665 samples used) :

All named entities :
 * precision = 0.779
 * recall    = 0.398
 * accuracy  = 0.665

Named entities followed by punctuation :
 * precision = 0.878
 * recall    = 0.332
 * accuracy  = 0.665

Named entities with interjection in neighborhood :
 * precision = 0.867
 * recall    = 0.079
 * accuracy  = 0.563
```
 
 
 #### Results with learning

``` 
Loading named entities dataset from file 'data/features_ne.csv'
Loading named entities dataset from file 'data/features_ne_punct.csv'
Loading named entities dataset from file 'data/features_ne_interj.csv'

Dimensions of datasets :
 * train : (1865, 57)
 * valid : (400, 57)
 * test  : (400, 57)

==================================================
                     SHELDON                      
==================================================

Training classifiers independently
Training model RandomForestClassifier 2
 * Precision : 81.74%
 * Recall    : 85.65%
 * Accuracy  : 80.75%
Training model XGBoost 1
 * Precision : 81.17%
 * Recall    : 84.35%
 * Accuracy  : 79.75%

==================================================
                     LEONARD                      
==================================================

Training classifiers independently
Training model RandomForestClassifier 2
 * Precision : 73.99%
 * Recall    : 59.26%
 * Accuracy  : 66.75%
Training model XGBoost 1
 * Precision : 73.45%
 * Recall    : 60.19%
 * Accuracy  : 66.75%

==================================================
                      PENNY                       
==================================================

Training classifiers independently
Training model RandomForestClassifier 2
 * Precision : 73.50%
 * Recall    : 50.29%
 * Accuracy  : 71.00%
Training model XGBoost 1
 * Precision : 73.79%
 * Recall    : 44.44%
 * Accuracy  : 69.50%

==================================================
                       RAJ                        
==================================================

Training classifiers independently
Training model RandomForestClassifier 2
 * Precision : 67.01%
 * Recall    : 50.78%
 * Accuracy  : 76.25%
Training model XGBoost 1
 * Precision : 67.05%
 * Recall    : 46.09%
 * Accuracy  : 75.50%

==================================================
                      HOWARD                      
==================================================

Training classifiers independently
Training model RandomForestClassifier 2
 * Precision : 70.63%
 * Recall    : 60.14%
 * Accuracy  : 76.00%
Training model XGBoost 1
 * Precision : 69.77%
 * Recall    : 60.81%
 * Accuracy  : 75.75%

Tests results saved to 'data/prediction_ne_test.csv'
```
