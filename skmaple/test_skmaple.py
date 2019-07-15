from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

X, y = make_classification()
rf = RandomForestClassifier(n_estimators=100)
lr = LogisticRegression(solver='lbfgs', multi_class='auto')

X_train, X_test, y_train, y_test = train_test_split(X, y)

maple = MAPLE(rf, lr)
maple.fit(X_train, y_train)
preds = maple.predict(X_test)
print('MAPLE:')
print(confusion_matrix(y_test, preds))

rf.fit(X_train, y_train)
preds = rf.predict(X_test)
print('Random Forest:')
print(confusion_matrix(y_test, preds))
