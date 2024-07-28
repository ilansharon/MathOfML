import numpy as np
import matplotlib.pyplot as plt

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
    gradient = []
    for i in range(len(par)):
        gradient.append(partial(func, par, xData, yData, i))
    return np.array(gradient)

# Time for school!
# Recommended values:
# decay1 = 0.9
# decay2 = 0.99
# shift = 1e-8
def learnViaAdam(func, par, step, decay1, decay2, shift, xData, yData): # So many arguments, doing something inneficiently but not sure what (use a class?)
    m = np.zeros(len(par))
    v = np.zeros(len(par))
    t = 0
    path = [par.copy()] # recording the path for visualization

    # this threshold needs elaboration, for now just going to hard code
    for _ in range(1000):
        t += 1
        g = gradient(func, par, xData, yData)
        m = decay1 * m + (1 - decay1) * g               #momentum
        v = decay2 * v + (1 - decay2) * g**2            #rmsprop
        mcor = m / (1 - decay1 ** t)                    #bias correction \/
        vcor = v / (1 - decay2 ** t)
        par -= step * mcor / (np.sqrt(vcor) + shift)    # parameter update
        path.append(par.copy())
    return par, np.array(path)                                    #resulting params and path






# Visualization: -------------------------
def funcToArr(func, length, par):
    f = []
    for i in range(length):
        f.append(func(par, i))
    return np.array(f)


def plotFinalFit(fit, xData, yData, par):
    fig, ax = plt.subplots()
    ax.scatter(xData, yData)
    ax.plot(funcToArr(fit, len(xData), par))
    ax.set_title('Fit and Real Data')
    plt.show()
    return

def animateFit(fit, xData, yData, path):
    fig, ax = plt.subplots()
    ax.scatter(xData, yData, label='Data')
    line, = ax.plot(xData, funcToArr(fit, len(xData), path[0]), color='red', label='Fit')
    for i in path:
        yfit = funcToArr(fit, len(xData), i)
        line.set_data(xData, yfit)
        plt.pause(0.01)


    ax.set_title('Fit and Real Data')
    plt.show()
    return

def plotError(par, path, xData, yData): #only plots a and b
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xCenter = par[0] #a
    yCenter = par[1] #b
    x = np.linspace(xCenter-500, xCenter+500, 10000)
    y = np.linspace(yCenter-500, yCenter+500, 10000)
    x, y = np.meshgrid(x, y)

    z = avgError(par, xData, yData)
    ax.plot_surface(x, y, z, cmap='viridis')

    scatter = None

    for i in path:

        if scatter:
            scatter.remove()

        z = avgError(i, xData, yData)
        ax.plot_surface(x, y, z, cmap='viridis')
        ax.scatter(i[0], i[1], z, cmap='viridis')
        plt.pause(0.1)

    ax.set_title('Error data')
    plt.show()
    return

def plotScalarLoss(path, xData, yData):
    y = []
    for i in path:
        y.append(int(avgError(i, xData, yData)))
    y = np.array(y)
    x = np.arange(1, len(path) + 1)
    plt.plot(x, y)
    plt.xlabel('Time')
    plt.ylabel('Loss')
    plt.title('Scalar Loss / Time')
    plt.show()



# init ---------------
def getInitParams():
    init = int(input('Would you like to input your own initial parameters? If yes, enter 1, if no, enter 0: '))
    if init == 0:

        defaults = [100, 0.02, 1, 420]
        return defaults

    if init == 1:
        print("Now inputting parameters:")
        print("function is of the form: (asin(bx + c) + d")
        p1 = float(input('Input parameter a: '))
        p2 = float(input('Input parameter b: '))
        p3 = float(input('Input parameter c: '))
        p4 = float(input('Input parameter d: '))
        initParams = [p1, p2, p3, p4]
        return initParams