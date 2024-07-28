import data
import methods
import os

print("would you like to use your own data, or in auto mode?")
print("***construction*** own data is WIP, must be in x,y simple format")
ownData = input("enter: '0' for auto, or 1 for your own: ")
if ownData == "1":
    filepath = input('input path to file').strip('"')
    dataset = data.load_data(filepath, 0)
else:
    cd = os.path.dirname(os.path.abspath(__file__))
    defaultpath = os.path.join(cd, 'defaultdata.csv')
    dataset = data.load_data(defaultpath, 0)
    print(dataset)
    dataset = data.clean_data(dataset)


X = dataset[dataset.columns[0]].values
Y = dataset[dataset.columns[1]].values


params = methods.getInitParams()
print()
print("your initial parameters are: ")
print(params)
print()

idealPar, path = methods.learnViaAdam(methods.avgError, params, 0.01, 0.9, 0.99, 1e-8, X, Y)
finalLoss = int(methods.avgError(idealPar, X, Y))

print(idealPar)
print("Your final average error is %d" % finalLoss)

methods.plotScalarLoss(path, X, Y)
methods.animateFit(methods.rawSin, X, Y, path)



