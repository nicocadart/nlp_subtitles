# nlp_subtitles

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
 * train : (1865, 33)
 * valid : (400, 33)
 * test  : (400, 33)

==================================================
                     SHELDON                      
==================================================

Training classifiers independently
Training model LogisticRegression
 * Precision : 82.78%
 * Recall    : 75.22%
 * Accuracy  : 76.75%
Training model RandomForestClassifier 2
 * Precision : 77.49%
 * Recall    : 77.83%
 * Accuracy  : 74.25%
Training model XGBoost 1
 * Precision : 76.96%
 * Recall    : 76.96%
 * Accuracy  : 73.50%

==================================================
                     LEONARD                      
==================================================

Training classifiers independently
Training model LogisticRegression
 * Precision : 70.56%
 * Recall    : 64.35%
 * Accuracy  : 66.25%
Training model RandomForestClassifier 2
 * Precision : 72.77%
 * Recall    : 64.35%
 * Accuracy  : 67.75%
Training model XGBoost 1
 * Precision : 70.37%
 * Recall    : 61.57%
 * Accuracy  : 65.25%

==================================================
                      PENNY                       
==================================================

Training classifiers independently
Training model LogisticRegression
 * Precision : 65.38%
 * Recall    : 39.77%
 * Accuracy  : 65.25%
Training model RandomForestClassifier 2
 * Precision : 63.27%
 * Recall    : 36.26%
 * Accuracy  : 63.75%
Training model XGBoost 1
 * Precision : 64.89%
 * Recall    : 35.67%
 * Accuracy  : 64.25%

==================================================
                       RAJ                        
==================================================

Training classifiers independently
Training model LogisticRegression
 * Precision : 66.67%
 * Recall    : 37.50%
 * Accuracy  : 74.00%
Training model RandomForestClassifier 2
 * Precision : 73.33%
 * Recall    : 42.97%
 * Accuracy  : 76.75%
Training model XGBoost 1
 * Precision : 73.44%
 * Recall    : 36.72%
 * Accuracy  : 75.50%

==================================================
                      HOWARD                      
==================================================

Training classifiers independently
Training model LogisticRegression
 * Precision : 68.13%
 * Recall    : 41.89%
 * Accuracy  : 71.25%
Training model RandomForestClassifier 2
 * Precision : 70.71%
 * Recall    : 47.30%
 * Accuracy  : 73.25%
Training model XGBoost 1
 * Precision : 67.37%
 * Recall    : 43.24%
 * Accuracy  : 71.25%

Tests results saved to 'data/prediction_ne_test.csv'
```