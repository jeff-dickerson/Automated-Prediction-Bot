# Automated Prediction Algorithm


Start by using Random Forest Classifier (best for non-linear data relationships) to training a bunch of individual decision trees with randomized parameters and than averaging the results from those decision trees.
The training test set was split up to avoid overfitting and data leakage

```
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

train = sp500.iloc[:-100]
test = sp500.iloc[-100]

predictors = ["Close", "Volume", "Open" , "High", "Low"]
model.fit(train[predictors], train["Target"])
```

The training test set was split up to avoid overfitting and data leakage

```
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(train[predictors], train["Target"])

test[predictors] = test[predictors].values.reshape(-1, len(predictors))
preds = model.predict(test[predictors])
```

This project is a predecessor to a much bigger undertaking that tackles building a multimodal model for financial services

