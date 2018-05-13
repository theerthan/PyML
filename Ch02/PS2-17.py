from numpy import mean
from math import sqrt

def cov(x,y,pxy):
    """Covariance of a joint distribution
    Defined as summation of x(i) times y(i) times P(X=x(i),Y=y(i)) over
    mean(x) times mean(y)
    """
    x_mean = mean(x)
    y_mean = mean(y)
    cov_l = [x[i]*y[j]*pxy[i][j] for i in range(len(x)) for j in range(len(y))]
    cov_val = sum(cov_l) - (x_mean * y_mean)
    return cov_val

def mystd(x,pX):
    """Std of a prob distribution
    Defined as vax(x) = summation of (x(i)-mu)^2 times P(X=x(i))
    And Std = Sq root of vax(x)
    """
    mu_x = mean(x)
    var_x_l = [(x[i]-mu_x)**2*pX[i] for i in range(len(x))]
    var_x = sum(var_x_l)
    return sqrt(var_x)

def corr(x,y,pX,pY,pxy):
    """Correlation of a joint distribution
    Defined as covariance of the joint distribution over
    the product of the std of the individual distributions
    """
    return cov(x,y,pxy)/(mystd(x,pX)*mystd(y,pY))

if __name__ == '__main__':
    X = [-1,0,1]
    Y = [-1,0,1]
    pXY = [[0,0,1/3],
           [0,1/3,0],
           [1/3,0,0]]
    pX = [1/3,1/3,1/3]
    pY = [1/3,1/3,1/3]

    print(cov(X,Y,pXY))
    print(corr(X,Y,pX,pY,pXY))

