import numpy as np
import math
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

dataset = datasets.fetch_california_housing()
print('loaded', dataset.data.shape, dataset.target.shape)
dataset_X = dataset.data[:442]
dataset_y = dataset.target[:442]
dataset_X_train = dataset_X[:362]
dataset_y_train = dataset_y[:362]
dataset_X_test = dataset_X[362:442]
dataset_y_test = dataset_y[362:442]
model = linear_model.LinearRegression()
model.fit(dataset_X_train, dataset_y_train)
X_train_bias = np.c_[np.ones(dataset_X_train.shape[0]), dataset_X_train]
weights_manual = np.linalg.pinv(X_train_bias.T.dot(X_train_bias)).dot(X_train_bias.T).dot(dataset_y_train)
pred_lib = model.predict(dataset_X_test)
pred_manual = np.c_[np.ones(dataset_X_test.shape[0]), dataset_X_test].dot(weights_manual)
rmse_lib = math.sqrt(mean_squared_error(dataset_y_test, pred_lib))
rmse_manual = math.sqrt(mean_squared_error(dataset_y_test, pred_manual))
print('rmse_lib', rmse_lib, 'rmse_manual', rmse_manual)
