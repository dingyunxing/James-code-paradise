import numpy as np
def GradientDescent(y, x):
    '''a functon to calculate the coefficient of each feature with gradient 
    descent
    return beta, which is a matrix of coefficients of all the control variables
    x, y are python lists
    '''
    l = len(x[0]) # how many features
    m = len(x) # how many rows of data
    beta = np.ones(l)
    x = np.array(x)
    y = np.array(y)
    
    # formular: beta = (X.T*X)-1 * X.T *Y
    #beta = np.array([ -3.67585642e+02,  -3.70182336e-02,   6.16163588e+00])
    
    diff = 0.001
    alpha = 0.01
    i = 0
    difference = 1000
    while not(difference <= diff or i >= 10000):
        #define loss function
        lossfunction = np.sum((x.dot(beta) - y) ** 2)/2/m
        #define the gradient
        gradient = (x.T.dot(x.dot(beta)-y))/m 
        #print(gradient)
        beta -= alpha * gradient
        #print(beta)
        difference = np.sum((x.dot(beta) - y) ** 2)/2/m - lossfunction
        print(difference)
        i += 1
    return beta




