import k_means_class as kmc
import numpy as np
# Example dataset (5 points, 2 features)
data = np.array([
    [1.0, 2.0],
    [1.5, 1.8],
    [5.0, 8.0],
    [8.0, 8.0],
    [1.0, 0.6]
])

# Initialize K-Means with 2 clusters
kmeans = kmc.k_means(2, data)

# Run the K-Means algorithm
kmeans.build()

# Output the final centroids and the cluster assignments
print("Centroids:\n", kmeans.centroids)
print("Cluster Assignments:", kmeans.nearscluster)
