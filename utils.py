import numpy as np
np.random.seed(2022)

"""encode data to one hot format"""
def encode_one_hot(data):
    X = [[*x]for x in data]
    X_oh = np.zeros((len(X), 1576, 26))
    for (ind, x) in enumerate(X_oh):
        x[np.arange(len(X[ind])),[ord(cha)-65 for cha in X[ind]]]=1
    X_oh = np.delete(X_oh,[1,14,20],2)
    return X_oh

"""random shuffle two arrays in unison"""
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]