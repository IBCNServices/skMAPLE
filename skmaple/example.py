from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from skmaple import MAPLE

X, y = make_regression()
rf = RandomForestRegressor(n_estimators=100)
lr = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y)

maple = MAPLE(rf, lr)
maple.fit(X_train, y_train)
preds = maple.predict(X_test)
print('MAPLE:')
print(mean_squared_error(y_test, preds))

rf.fit(X_train, y_train)
preds = rf.predict(X_test)
print('Random Forest:')
print(mean_squared_error(y_test, preds))
