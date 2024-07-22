import numpy as np
import matplotlib as plt


def rawSin(par, x):
    return par[0] * np.sin((par[1] * x) + par[2]) + par[3]

def sqErrorSin(par, x, y):
   return ((par[0] * np.sin((par[1] * x) + par[2]) + par[3]) - y)**2

def avgError(par, xData, yData):
    total = 0
    n = len(xData)
    for i in range(n):
        total += sqErrorSin(par, xData[i], yData[i])
    avg = total / n
    return avg


# Compute partial derivative d/dpar[i] using finite difference method with step h
def partial(func, par, xCur, yCur, h, i):
    upPar = par.copy()
    upPar[i] += h
    downPar = par.copy()
    downPar[i] -= h
    partial = (func(upPar, xCur, yCur) - func(downPar, xCur, yCur)) / (2 * h)
    return partial

def gradient(func, par, xCur, yCur, h):
    gradient = np.array([])
    for i in range(len(par)):
        gradient = gradient.append(partial(func, par, xCur, yCur, h, i))
    return gradient
