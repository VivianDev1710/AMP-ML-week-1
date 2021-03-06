import numpy as np
from sigmoid import sigmoid


def costFunction(theta, X, y):
    '''returns cost for theta, X and y
    np.log(a)==> returns array with elementwise log on array a
    use the sigmoid function that's being imported above 
    '''
    m = y.size
    h =sigmoid(np.dot(X,theta))
    cost=(y*np.log(h)+(1-y)*(1+np.log(h)))
    J =-(1/m)*np.sum(cost)

    if np.isnan(J):
        return(np.inf)
    return(J)


def gradient(theta, X, y):
	'''' calculate gradient descent for logistic regression'''
	m = y.size
	theta=theta.reshape(-1,1)
	h = sigmoid(np.dot(X,theta))
	alpha=0.01
	grad = alpha*np.sum(np.dot((np.transpose(X),(h-y))))
	return(grad.flatten())			# returns copy of array in one dimension
