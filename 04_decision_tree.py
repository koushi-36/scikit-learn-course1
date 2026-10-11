"""Decision Tree classifier on the Breast Cancer Wisconsin dataset."""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Decision Tree")
print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
print(classification_report(y_test, predictions, target_names=data.target_names))
print("Feature importance (top 5):")
for idx in model.feature_importances_.argsort()[::-1][:5]:
    print(f"- {data.feature_names[idx]}: {model.feature_importances_[idx]:.3f}")
