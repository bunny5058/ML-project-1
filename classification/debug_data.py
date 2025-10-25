from sklearn.datasets import fetch_openml
import numpy as np

# Load the data
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target']

# Convert target to numeric
y = y.astype(int)

# Split the data
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Create binary target
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

print('y_train unique values:', np.unique(y_train))
print('y_train_5 unique values:', np.unique(y_train_5))
print('Number of True in y_train_5:', np.sum(y_train_5))
print('Number of False in y_train_5:', len(y_train_5) - np.sum(y_train_5))
print('Percentage of 5s in training data:', np.sum(y_train_5) / len(y_train_5) * 100)

# Also check the data types and shapes
print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('y_train_5 shape:', y_train_5.shape)
print('X_train type:', type(X_train))
print('y_train type:', type(y_train))
print('y_train_5 type:', type(y_train_5))
