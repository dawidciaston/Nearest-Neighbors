import numpy as np
from typing import Optional, Union, Tuple
from collections import Counter

class NearestNeighbors:

    def __init__(self, n_neighbors: int = 5, metric: str = 'euclidean', p: int = 3):
        # setting up starting params here
        # n_neighbors is k
        self.n_neighbors = n_neighbors
        # metric is how we calculate distance (euclidean etc)
        self.metric = metric
        # p is needed only for minkowski
        self.p = p
        # training data will go here but empty for now
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NearestNeighbors':
        # this is lazy learning, i dont calculate anything here
        # just saving data to variables and that's it
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def _calculate_distance(self, point: np.ndarray) -> np.ndarray:
        # function to calc distance from one point to the rest
        
        # substracting point from whole matrix at once
        diff = np.abs(self.X_train - point)
        
        if self.metric == 'euclidean':
            # standard pythagoras - sqrt of sum of squares
            return np.sqrt(np.sum(diff ** 2, axis=1))
        
        elif self.metric == 'manhattan':
            # sum of modules, like walking in a city
            return np.sum(diff, axis=1)
        
        elif self.metric == 'minkowski':
            # formula with power p
            return np.power(np.sum(diff ** self.p, axis=1), 1 / self.p)
        
        else:
            # if someone types nonsense in metric name then throw error
            raise ValueError(f"Unknown metric: {self.metric}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        # main function that predicts classes
        
        # first checking if i even put training data in
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model not fitted!")
            
        X = np.array(X)
        predictions = []
        # small number so i dont divide by zero if distance is 0
        epsilon = 1e-6 

        # looping through every point i need to check
        for i in range(len(X)):
            # 1. calculating how far it is to all training points
            distances = self._calculate_distance(X[i])
            
            # 2. sorting and taking k nearest indices
            # argsort returns indices sorted from smallest distance
            k_indices = np.argsort(distances)[:self.n_neighbors]
            
            # getting actual distances and labels for these k neighbors
            k_distances = distances[k_indices]
            k_labels = self.y_train[k_indices]
            
            # 3. weighting votes (closer means more important)
            weights = 1 / (k_distances + epsilon)
            
            # 4. summing weights for each class
            class_votes = {}
            for label, weight in zip(k_labels, weights):
                # if key doesnt exist give 0 and add weight
                class_votes[label] = class_votes.get(label, 0) + weight
                
            # picking class that got most points
            predicted_class = max(class_votes, key=class_votes.get)
            predictions.append(predicted_class)
            
        return np.array(predictions)