import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Load the McDonald's dataset
mcdonalds = pd.read_csv("mcdonalds.csv")

# Exploring Data


MD_x = mcdonalds.iloc[:, 1:12].replace({"Yes": 1, "No": 0}).values
MD_pca = PCA()
MD_pca.fit(MD_x)

# Printing Summary
print("Summary of PCA:")
print("Explained Variance Ratio:", MD_pca.explained_variance_ratio_)
print("Singular Values:", MD_pca.singular_values_)
print("Components (Eigenvectors):", MD_pca.components_)

# Visualization
plt.figure(figsize=(8, 6))
plt.scatter(MD_pca.components_[0], MD_pca.components_[1], color='grey')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Principal Component Analysis')
plt.grid(True)
plt.show()

# Extracting Segments

# : Using K-means
kmeans_models = {}
for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=1234)
    kmeans.fit(MD_x)
    kmeans_models[k] = kmeans

# Visualizing the results
plt.figure(figsize=(10, 5))
plt.plot(list(kmeans_models.keys()), [model.inertia_ for model in kmeans_models.values()], marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Within-cluster Sum of Squares)')
plt.title('Elbow Method for KMeans')
plt.grid(True)
plt.show()

# Using Gaussian Mixture Models
gmm_models = {}
for k in range(2, 9):
    gmm = GaussianMixture(n_components=k, random_state=1234)
    gmm.fit(MD_x)
    gmm_models[k] = gmm

# Visualizing the results
plt.figure(figsize=(10, 5))
for k, model in gmm_models.items():
    plt.plot(k, model.bic(MD_x), marker='o', label=f'{k} components')
plt.xlabel('Number of Components (k)')
plt.ylabel('BIC Score')
plt.title('BIC Score for Gaussian Mixture Models')
plt.legend()
plt.grid(True)
plt.show()

# Profiling Segments

# Hierarchical clustering
MD_vclust = linkage(MD_x.T)
plt.figure(figsize=(10, 6))
dendrogram(MD_vclust, orientation='top', labels=mcdonalds.columns[1:12], leaf_font_size=10)
plt.title('Hierarchical Clustering Dendrogram')
plt.ylabel('Distance')
plt.show()

# Describing Segments

# Visualizing segment distribution by Like and segment number
plt.figure(figsize=(10, 6))
k4 = kmeans_models[4].labels_
mosaicplot = pd.crosstab(index=k4, columns=mcdonalds['Like']).apply(lambda r: r/r.sum(), axis=1).T.plot(kind='bar', stacked=True)
plt.xlabel('Segment Number')
plt.ylabel('Like Distribution')
plt.title('Segment Distribution by Like')
plt.legend(title='Like')
plt.show()

# Selecting Target Segments
visit_frequency = mcdonalds.groupby(k4)['VisitFrequency'].mean()
like = mcdonalds.groupby(k4)['Like.n'].mean()
female = mcdonalds.groupby(k4)['Gender'].apply(lambda x: (x == 'Female').mean())

plt.figure(figsize=(10, 6))
plt.scatter(visit_frequency, like, s=10*female, c=k4, cmap='viridis', alpha=0.5)
for i, txt in enumerate(range(1, 5)):
    plt.annotate(txt, (visit_frequency[i], like[i]))
plt.xlabel('Mean Visit Frequency')
plt.ylabel('Mean Like Score')
plt.title('Target Segments')
plt.colorbar(label='Segment Number')
plt.grid(True)
plt.show()
