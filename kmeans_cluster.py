'''
This is a build project for Lambda School CS Unit.
I am implementing a k-means clustering algorithm from scratch.
'''

# Imports
import matplotlib.pyplot as plt
import numpy as np
import random


# Distance between two points function using Euclidean geometry
def distance(x, y):
    return np.sqrt(np.sum((x-y)**2))

# Distance sqaured - way to punish clusters that are very separated
def distance_squared(x, y):
    return np.sum((x-y)**2)


# K-means class
class KMeans:
    # Initialization
    def __init__(self, k_clusters=2, max_iters=50, n_runs=5, plot=False):
        self.k_clusters = k_clusters
        self.max_iters = max_iters
        self.n_runs = n_runs

        # Storage for the centroid of each cluster
        self.centroids = []
        # Storage for clusters. Need an empty list for each of k_clusters
        self.clusters = [[] for num in range(self.k_clusters)]

        # Storage for the punishments, best centroids, and best grouping
        # for all of the n_runs
        self.punishments = []
        self.best_centroids = []
        self.best_grouping = []
        

    # Fit Method
    def fit(self, X):
        # Take in the intital data and assign clusters.
        # Interate until convergence or max_iters reached.
        # Repeat process n_runs times
        # Return the best fit
        
        # Incoming data
        self.X = X
        # Shape of data: samples = how many; features = dimensions
        self.samples, self.features = X.shape


        # Run through as many iterations as dictated by n_runs looking
        # for the best results.
        for _ in range(self.n_runs):
            # Randomly pick centroids from X
            self.centroids = random.sample(list(self.X), self.k_clusters)
            # Replace centroids with indices for the points
            self.centroids = [list(X[np.where(X == self.centroids[i])[0][0]]) for i in range(len(self.centroids))]


            # Iterate through finding the best clusters
            for _ in range(self.max_iters):
                # Placeholder for clusters in the prior run for use later
                prior_clusters = self.clusters
                # Update each point to closest centroid (updating the clusters)
                self.clusters = self.assign_clusters(self.centroids)

                # This is for plotting in the midst of running the fit method
                # if self.plot:
                #     self.plot()

                # Save prior centroids for use later, if needed
                prior_centroids = self.centroids
                # Update each centroid's location to center of cluster
                self.centroids = self.update_centroids(self.clusters)

                # if self.plot:
                #     self.plot()

                # Check if centroids have converged (not moved)
                distances = [distance(prior_centroids[i], self.centroids[i]) for i in range(self.k_clusters)]
                # If they have converged then get out of the loop
                if sum(distances) == 0:
                    break
        
        # These are the assignments for each point in X
        grouping = np.zeros(self.samples)
        for cluster_index, cluster in enumerate(self.clusters):
            for sample_index in cluster:
                grouping[sample_index] = cluster_index
        
        # Create a punishment measurement for clusters that are spread out as opposed
        # to tightly packed
        current_punishments = self.punisher(self.clusters, grouping, self.centroids)


        # Check if this iteration is the best one yet
        if self.best_centroids == []:
            # Set the standard with the first run
            self.best_centroids = self.centroids
            self.punishments = current_punishments
            self.best_grouping = grouping
        else:
            # If the current run has a better (smaller) score than the previous best...
            if sum(current_punishments) < sum(self.punishments):
                # Save the current centroids and grouping
                self.best_centroids = self.centroids
                self.best_grouping = grouping

        # These are print statements used in testing
        # print(self.clusters)
        # print(self.centroids)
        # print(current_punishments)
        # print(self.best_centroids)
        # print('Grouping:', grouping)
        # print('Best Grouping:', self.best_grouping)


        # Return the 'labels' or group created for the best centroids
        return self.best_grouping



    # Predict Method
    def predict(self, X):
        # Take in new data
        # Decide which clusters the new data should belong to given the fit performed
        # Return grouped data

        # Incoming data
        self.X = X

        # Placeholder for predictions
        predictions = np.zeros(len(self.X))

        # Loop through data in X
        for point_index, point in enumerate(self.X):
            # For each item, check for shortest distance between it and the centroids
            distances = []
            for centroid_index, centroid in enumerate(self.best_centroids):
                distances.append(distance(point, centroid))
                predictions[point_index] = distances.index(min(distances))

        # Return prediction of which group each datapoint will belong to
        return predictions



    # Helper functions
    # Assign points to clusters
    def assign_clusters(self, centroids):
        clusters = [[] for num in range(self.k_clusters)]
        for i, point in enumerate(self.X):
            centroid_index = self.nearest_centroid(point, centroids)
            clusters[centroid_index].append(i)
        return clusters

    # Find nearest centroid
    def nearest_centroid(self, point, centroids):
        distances = [distance(point, centroid) for centroid in centroids]
        nearest = np.argmin(distances)
        return nearest

    # Update centroids
    def update_centroids(self, clusters):
        centroids = np.zeros((self.k_clusters, self.features))
        for i, cluster in enumerate(clusters):
            cluster_location = np.mean(self.X[cluster], axis=0)
            centroids[i] = cluster_location
        return centroids

    # Distance Squared Checker
    def punisher(self, clusters, grouping, centroids):
        clusters_mean = []
        for i in range(len(clusters)):
            clusters_i_total = 0
            for point_index in clusters[i]:
                clusters_i_total += distance_squared(self.X[point_index], centroids[i])
            clusters_mean.append(clusters_i_total/len(clusters[i]))
        return clusters_mean
        

    
    # Plot graphs as centroids and clusters change
    def plot(self):
        fig, ax = plt.subplots(figsize=(12,8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)
        
        for point in self.centroids:
            ax.scatter(*point, marker='x', color='black', linewidth=2)
        
        plt.show()
