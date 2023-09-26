# Author: Created by Halowai on 25/9/2023
# Project: WaiCheng_COMP257_Assignment1
# File Name: question2

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# 1. Generate Swiss roll dataset.
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)


# 2. Plot the resulting generated Swiss roll dataset.
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.jet)
ax.set_title("Swiss Roll Dataset")
plt.show()


# 3. Use Kernel PCA (kPCA) with linear kernel, a RBF kernel, and a sigmoid kernel.
# Applying kPCA with different kernels
lin_pca = KernelPCA(n_components=2, kernel="linear", fit_inverse_transform=True)
rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04, fit_inverse_transform=True)
sigmoid_pca = KernelPCA(n_components=2, kernel="sigmoid", gamma=0.001, coef0=1, fit_inverse_transform=True)
X_lin = lin_pca.fit_transform(X)
X_rbf = rbf_pca.fit_transform(X)
X_sigmoid = sigmoid_pca.fit_transform(X)


# 4. Plot the kPCA results of applying the linear kernel, a RBF kernel, and a sigmoid kernel.
# Plotting kPCA results
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Linear kernel
axes[0].scatter(X_lin[:, 0], X_lin[:, 1], c=t, cmap=plt.cm.jet)
axes[0].set_title("Linear kernel")

# RBF kernel
axes[1].scatter(X_rbf[:, 0], X_rbf[:, 1], c=t, cmap=plt.cm.jet)
axes[1].set_title("RBF kernel")

# Sigmoid kernel
axes[2].scatter(X_sigmoid[:, 0], X_sigmoid[:, 1], c=t, cmap=plt.cm.jet)
axes[2].set_title("Sigmoid kernel")

plt.tight_layout()
plt.show()


# 5. Using kPCA and a kernel of your choice, apply Logistic Regression for classification.
# Use GridSearchCV to find the best kernel and gamma value for kPCA in order to get the best classification accuracy at the end of the pipeline.
# Creating a pipeline
clf = Pipeline([
    ("kpca", KernelPCA(n_components=2)),
    ("log_reg", LogisticRegression())
])

# Parameters for GridSearchCV
param_grid = [{
    "kpca__gamma": np.linspace(0.03, 0.05, 10),
    "kpca__kernel": ["linear", "rbf", "sigmoid"]
}]
grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X, t > 6.9)  # Classifying based on the t value for demonstration
# Printing best parameters
print("Best parameters found by GridSearchCV:", grid_search.best_params_)


# 6. Plot the results from using GridSearchCV in (5).
# Plotting the results
X_grid = grid_search.best_estimator_["kpca"].transform(X)
plt.scatter(X_grid[:, 0], X_grid[:, 1], c=t > 6.9, cmap="jet")
plt.title("Best kPCA with GridSearchCV")
plt.show()
