"""Gaussian Naive Bayes classification on the Iris dataset."""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)
model = GaussianNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Gaussian Naive Bayes")
print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions, target_names=data.target_names))
