import data
import methods

filepath = input('input path to file')
dataset = data.load_data(filepath)
dataset = data.clean_data(dataset)
X = dataset[dataset.columns[0]].values
Y = dataset[dataset.columns[1]].values

print (X)
print (Y)
params = methods.getInitParams()
print()
print("your initial parameters are: ")
print(params)
print()

idealPar, path = methods.learnViaAdam(methods.avgError, params, 0.001, 0.9, 0.99, 1e-8, X, Y)
print(idealPar)
methods.animateFit(methods.rawSin, X, Y, path)
#methods.plotScalarLoss(path, X, Y)


