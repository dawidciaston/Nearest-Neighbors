import pytest
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
from sklearn.datasets import make_blobs
from nn_project.impl import NearestNeighbors as MyKNN


def test_slide_requirements_parity():
    """
    Checks **Logic & Consistency** parity for all required metrics
    against a reference implementation (Scikit-Learn) on an N=200 dataset.
    Replaces and extends older 'parity' and 'metrics' tests.
    """
    # Generate N=200 data
    x, y = make_blobs(n_samples=200, centers=3, n_features=4, random_state=42)

    # List of test cases from the slide: (Metric, k, p)
    test_cases = [
        ('euclidean', 3, 2),
        ('manhattan', 5, 1),
        ('minkowski', 1, 3)
    ]

    for metric, k, p in test_cases:
        # 1. Scikit-Learn (Reference/Ground Truth)
        sk_model = SklearnKNN(n_neighbors=k, weights='distance', metric=metric, p=p)
        sk_model.fit(x, y)
        sk_preds = sk_model.predict(x)

        # 2. Custom implementation
        my_model = MyKNN(n_neighbors=k, metric=metric, p=p)
        my_model.fit(x, y)
        my_preds = my_model.predict(x)

        # 3. Verification
        assert np.array_equal(my_preds, sk_preds), \
            f"Logical error for configuration: {metric}, k={k}, p={p}"


def test_errors_and_validation():
    """
    Checks error handling and validation (required for 100% Code Coverage).
    """
    model = MyKNN()

    # Error 1: Attempting prediction before training
    with pytest.raises(ValueError, match="Model not fitted"):
        model.predict([[0, 0]])

    # Error 2: Unknown metric provided
    model.fit(np.array([[0]]), np.array([0]))
    model.metric = 'cosmic_metric'
    # The actual distance calculation method should raise the error
    with pytest.raises(ValueError, match="Unknown metric"):
        # Calling an internal method that uses the metric attribute
        model._calculate_distance(np.array([[0]]))