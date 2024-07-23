import data
import methods

filepath = input('input path to file')
dataset = data.load_data(filepath)
dataset = data.clean_data(dataset)
X = dataset[dataset.columns[0]].values
Y = dataset[dataset.columns[1]].values

params = methods.getInitParams()
print()
print("your initial parameters are: ")
print(params)
print()

idealPar, path = methods.learnViaAdam(methods.avgError, params, 0.0001, 0.9, 0.99, 1e-8, X, Y)
print(idealPar)
methods.plotFit(methods.rawSin, X, Y, idealPar)
methods.plotError(params, path, X, Y)

