import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=64, n_informative=64,
                           n_redundant=0, n_classes=2, n_clusters_per_class=1)

# Filter data to include only digits 0 and 1
X_filtered = []
y_filtered = []
for i in range(len(y)):
    if y[i] == 0 or y[i] == 1:
        X_filtered.append(X[i])
        y_filtered.append(y[i])

X_filtered = np.array(X_filtered)
y_filtered = np.array(y_filtered)

# Visualize the first few samples
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))

for i, ax in enumerate(axes.flat):
    ax.imshow(X_filtered[i].reshape(8, 8), cmap='gray')
    ax.set_title(f'Digit: {y_filtered[i]}')
    ax.axis('off')

plt.tight_layout()
plt.show()

print('ljsfdlsfjds')