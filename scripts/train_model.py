from joblib import dump, load

from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

RANDOM_STATE = 123

X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE)

model = GradientBoostingClassifier(random_state=RANDOM_STATE)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

train_score = f1_score(y_train, model.predict(X_train), average="weighted")
test_score = f1_score(y_test, model.predict(X_test), average="weighted")

print(f"Train f1-score: {train_score}")
print(f"Test f1-score: {test_score}")

model_path = "../models/iris_v1.joblib"
dump(model, model_path)
model = load(model_path)

print(model.predict([[1, 1, 1, 1]]))