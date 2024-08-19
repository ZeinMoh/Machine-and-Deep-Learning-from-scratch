import numpy as np
import random
import math

class k_means:
    def __init__(self, k, po):
        self.k = k
        self.points = po.shape[0]
        self.pro = po.shape[1]
        self.pointcor = po
        self.centroids = po[np.random.choice(po.shape[0], k, replace=False)]  
        self.nearscluster = np.zeros(self.points, dtype=int)
        
    def dcal(self,x, y):
        return np.linalg.norm(x - y)

    def choose_cluster(self):
        for i in range(self.points):
            mind = float('inf')
            minc = -1
            for c in range(self.k):
                if i==c:
                    mind=0.0
                    minc=i
                else:
                    d = self.dcal(self.pointcor[i], self.centroids[c])
                    if d < mind:
                        mind = d
                        minc = c
            self.nearscluster[i] = minc

    def update_centroids(self):
        new_centroids = np.zeros_like(self.centroids)
        counts = np.zeros(self.k)
        
        for i in range(self.points):
            cluster = self.nearscluster[i]
            new_centroids[cluster] += self.pointcor[i]
            counts[cluster] += 1
        
        for c in range(self.k):
            if counts[c] > 0:
                new_centroids[c] /= counts[c]
        
        self.centroids = new_centroids
    
    def build(self, max_iters=10000):
        for _ in range(max_iters):
            old_centroids = self.centroids.copy()
            self.choose_cluster()
            self.update_centroids()
            if np.allclose(old_centroids, self.centroids):
                break

