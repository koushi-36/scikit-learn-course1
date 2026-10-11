"""Compare common classification metrics with a simple baseline model."""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Model Evaluation")
print("Accuracy :", round(accuracy_score(y_test, predictions), 3))
print("Precision:", round(precision_score(y_test, predictions), 3))
print("Recall   :", round(recall_score(y_test, predictions), 3))
print("F1 score :", round(f1_score(y_test, predictions), 3))
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
print("5-fold CV accuracy mean:", round(cv_scores.mean(), 3))
