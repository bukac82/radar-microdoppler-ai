# Classical ML Module

Traditional machine learning classifiers applied to hand-crafted micro-Doppler features.

## Files

| File | Description |
|------|-------------|
| `train.py` | Train SVM, Random Forest, XGBoost, k-NN classifiers |
| `evaluate.py` | Evaluate trained models with confusion matrix & classification report |
| `hyperparameter_search.py` | GridSearchCV / RandomizedSearchCV tuning |
| `models/` | Saved `.joblib` model artifacts |

## Pipeline

```
IQ signals
  └─> Preprocessing.features.extract_features()   # 14-D feature vector
      └─> Preprocessing.normalize (StandardScaler)
          └─> SVM / RF / XGBoost / k-NN
              └─> Evaluation.metrics
```

## Usage

```python
from Classical_ML.train import train_svm, train_random_forest
from Dataset.loader import load_dataset, get_train_test_split
from Preprocessing.features import extract_features

df = load_dataset(nrows=5000)
X_iq_train, X_iq_test, y_train, y_test = get_train_test_split(df)

X_train = extract_features(X_iq_train)
X_test  = extract_features(X_iq_test)

model = train_svm(X_train, y_train)
```
