Nearest Neighbors: Custom Implementation Project
================================================

A custom implementation of the K-Nearest Neighbors (KNN) algorithm, developed to understand vectorized distance calculations, lazy learning mechanics, and benchmark performance against the standard **Scikit-Learn** library.

Project Overview
================
This project implements KNN from scratch using a vectorized brute-force approach ($O(N^2)$) utilizing **NumPy Broadcasting** and compares it with Scikit-Learn's optimized version ($O(N \log N)$ using KD-Trees).

**Key Features:**
* **Vectorized Engine:** Full replacement of Python loops with optimized NumPy matrix operations for distance calculation.
* **Weighted Voting:** Implements **Inverse Distance Weighting** ($w = 1 / (d + \epsilon)$) to prioritize closer neighbors.
* **Multiple Metrics:** Support for **Euclidean**, **Manhattan**, and **Minkowski** distance metrics.
* **Lazy Learning:** No explicit training phase; computation is deferred until the prediction stage.
* **Scalability Analysis:** Generates a performance graph comparing execution time vs. dataset size on synthetic and real data.
* **Unit Tests:** Comprehensive tests ensuring algorithmic correctness and 100% parity with industry standards.

Installation
============
To install the project in editable mode (required for imports to work):

    pip install -e .

Usage
=====
Run Tests
------------
To verify the correctness of the algorithm (100% Parity Check against Scikit-Learn):

    pytest

Results Summary
===============
* **Accuracy:** The custom implementation achieves **100% agreement** of predictions with the Scikit-Learn library on both synthetic and Digits datasets.
* **Performance:** The vectorized implementation is **faster** for small datasets ($N < 1000$) due to low computational overhead. However, at larger scales, Scikit-Learn becomes faster due to optimized tree structures.