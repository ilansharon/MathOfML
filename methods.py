import numpy as np
import matplotlib as plt

# Actual fit function
def rawSin(par, x):
    return par[0] * np.sin((par[1] * x) + par[2]) + par[3]

# Squared error of fit function to actual data
def sqErrorSin(par, x, y):
   return ((par[0] * np.sin((par[1] * x) + par[2]) + par[3]) - y)**2

# Average of this error over all x, y
def avgError(par, xData, yData):
    total = 0
    n = len(xData)
    for i in range(n):
        total += sqErrorSin(par, xData[i], yData[i])
    avg = total / n
    return avg


# Compute partial derivative d/dpar[i] using finite difference method with step h
# func should be avgError
# i should be index of parameter with which the partial is taken with respect to

def partial(func, par, xData, yData, i):
    h = 1e-5
    upPar = par.copy()
    upPar[i] += h
    downPar = par.copy()
    downPar[i] -= h
    partial = (func(upPar, xData, yData) - func(downPar, xData, yData)) / (2 * h)
    return partial

# Combines partials of each element of par into gradient vector
def gradient(func, par, xData, yData):
    gradient = np.array([])
    for i in range(len(par)):
        gradient = gradient.append(partial(func, par, xData, yData, i))
    return gradient

# Time for school!
# Recommended values:
# decay1 = 0.9
# decay2 = 0.99
# shift = 1e-8
def learnViaAdam(func, par, step, decay1, decay2, shift, xData, yData): # So many arguments, doing something inneficiently but not sure what (use a class?)
    m = np.zeroes(len(par))
    v = np.zeroes(len(par))
    t = 0
    # this threshold needs elaboration, for now just going to hard code
    for _ in range(1000):
        t += 1
        g = gradient(func, par, xData, yData)
        m = decay1 * m + (1 - decay1) * g               #momentum
        v = decay2 * v + (1 - decay2) * g**2            #rmsprop
        mcor = m / (1 - decay1 ** t)                    #bias correction \/
        vcor = v / (1 - decay2 ** t)
        par -= step * mcor / (np.sqrt(vcor) + shift)    # parameter update
    return par                                          #resulting params