# MAPLE: Model Agnostic suPervised Local Explanations

by Gregory Plumb, Denali Molitor and Ameet S. Talwalkar ([http://papers.nips.cc/paper/7518-model-agnostic-supervised-local-explanations]([paper]) [https://blog.ml.cmu.edu/2019/07/13/towards-interpretable-tree-ensembles/]([blog]))


## How to use

### 1. Create an ensemble classifier (that has an `apply` method which returns the leaf indices of the different predictions) and a linear model

```python3
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=200, max_features=0.5, min_samples_leaf=10)
lr = Ridge(alpha=0.001)
```

### 2. Create the MAPLE object and pass along the ensemble and linear model
```python3
from skmaple import MAPLE
maple = MAPLE(rf, lr)
```

### 3. Fit and predict
```python3
maple.fit(X_train, y_train)
preds = maple.predict(X_test)
```

### Check out `skmaple/example.py` for an example!

## Reproducing accuracy experiments

In order to compare MAPLE to the linear model and ensemble method (which it uses), you can run `experiments/run.py`. Currently, it will only do 1 run per dataset, but this can easily be adapted (change the range value in the script).