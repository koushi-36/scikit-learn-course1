"""Agglomerative (hierarchical) clustering on a small built-in dataset."""
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

data = load_iris()
X = StandardScaler().fit_transform(data.data)

model = AgglomerativeClustering(n_clusters=3, linkage="ward")
labels = model.fit_predict(X)

print("Hierarchical Clustering")
print("Number of samples:", len(X))
print("Cluster sizes:", {int(label): int((labels == label).sum()) for label in set(labels)})
print("Silhouette score:", round(silhouette_score(X, labels), 3))
print("Note: cluster labels are group IDs, not the original flower classes.")
