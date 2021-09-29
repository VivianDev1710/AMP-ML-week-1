import numpy as np
def featureNormalization(X):
    """
    Take in numpy array of X values and return normalize X values,
    the mean and standard deviation of each feature

    np.mean(A)==> returns mean of all the elements of A

    np.std(A)==> returns the standard deviation of the elements of A
    """
    mean=np.hstack(np.mean(X[:,0]),np.mean(X[:,1]),np.mean(X[:,2]))
    std=np.hstack(np.std(X[:,0]),np.std(X[:,1]),np.std(X[:,2]))
    
    X_norm = (X - mean)/std
    
    return X_norm
