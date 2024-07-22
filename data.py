import pandas as pd

def load_data(filename):
    data = pd.read_csv(filename)
    return data
