import numpy as np
from pandas import DataFrame, Series
import pandas as pd

def convert_housing_to_quarters():
    housing_df = pd.read_csv('City_Zhvi_AllHomes.csv')
    #get indices of columns before 2000-01
    start = housing_df.columns.get_loc('1996-04')
    end = housing_df.columns.get_loc('2000-01')
    #remove unwanted columns from DataFrame
    housing_df = housing_df.drop(housing_df.columns[start:end], axis=1)
    #create a DataFrame for the date columns
    months_df = housing_df.iloc[:,6:]
    #transpose and reindex so months are in a column
    months_df = months_df.T
    months_df = months_df.reset_index()
    #convert months to quarters
    months_df['index'] = pd.PeriodIndex(pd.to_datetime(months_df['index']), freq='q')
    #calculate mean and transpose
    quarters_df = months_df.groupby('index').mean().T.round(2)
    #replace months columns in housing_df with quarters_df
    housing_df = housing_df.iloc[:,:6].join(quarters_df)
    #save to CSV
    housing_df.to_csv('quarterly_mean_housing_prices.csv')
