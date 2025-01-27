import pandas as pd
import datetime as dt
import os

def load_data(filename, skip):
    filename = os.path.abspath(filename)
    data = pd.read_csv(filename, skiprows=skip)
    return data

#for specific dataset
def clean_data(df):
    dateCol = df.columns[0]
    df[dateCol] = pd.to_datetime(df[dateCol])
    startTime = df[dateCol].min()
    timeSinceStart = df[dateCol] - startTime
    df[dateCol] = timeSinceStart.dt.total_seconds() / 60

    return df


