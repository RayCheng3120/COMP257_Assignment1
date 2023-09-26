# Author: Created by Halowai on 24/9/2023
# Project: WaiCheng_COMP257_Assignment1
# File Name: question1

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA, IncrementalPCA
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)


# 1. Retrieve and load the mnist_784 dataset of 70,000 instances.
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"].to_numpy(), mnist["target"].to_numpy()  # Convert to NumPy array
print("Dataset loaded successfully!")


# 2. Displaying each digit (0 to 9)
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i in range(10):
    ax = axes[i // 5, i % 5]
    ax.imshow(X[y == str(i)][0].reshape(28, 28), cmap="gray")
    ax.axis("off")
    ax.set_title(f"Digit: {i}")
plt.tight_layout()
plt.show()


# 3. Use PCA to retrieve the first and second principal component and output their explained variance ratio.
# Applying PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
# Displaying explained variance ratio
print("Explained variance ratio:", pca.explained_variance_ratio_)


# 4. Plot the projections of the first and second principal component onto a 1D hyperplane.
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], np.zeros_like(X_pca[:, 1]), c=y.astype(int), cmap="jet", alpha=0.5)
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component (Zeroed out)")
plt.colorbar()
plt.show()


# 5. Use Incremental PCA to reduce the dimensionality of the MNIST dataset down to 154 dimensions.
n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X, n_batches):
    inc_pca.partial_fit(X_batch)
X_reduced = inc_pca.transform(X)


# 6. Display the original and compressed digits from (5).
# Inverse transform to get the compressed version
X_recovered = inc_pca.inverse_transform(X_reduced)

# Displaying original and compressed digits
fig, axes = plt.subplots(2, 10, figsize=(10, 3))
for i in range(10):
    # Original digit
    ax = axes[0, i]
    ax.imshow(X[y == str(i)][0].reshape(28, 28), cmap="gray")
    ax.axis("off")
    ax.set_title(f"{i}")

    # Compressed digit
    ax = axes[1, i]
    ax.imshow(X_recovered[y == str(i)][0].reshape(28, 28), cmap="gray")
    ax.axis("off")

axes[0, 0].set_ylabel("Original")
axes[1, 0].set_ylabel("Compressed")
plt.tight_layout()
plt.show()
