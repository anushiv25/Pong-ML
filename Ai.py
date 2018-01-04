from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pickle

regr = linear_model.LogisticRegression(C=1e8)
trainX = []
trainY = []
with open('trainX.pkl', 'rb') as f:
    trainX = pickle.load(f)
with open('trainY.pkl', 'rb') as f:
    trainY = pickle.load(f)
tx = np.array(trainX)
print(np.array(trainY).astype('int'))
regr.fit(tx, np.array(trainY).astype('int'))
print(regr.coef_)
with open('weight.pkl', 'wb') as f:
    pickle.dump(regr.coef_, f)
