import numpy as np
import pandas as pd

def get_recessions():
    gdp_df = pd.read_csv('quarterly_gdp_2000-2017.csv')
    #calculate the change in gdp from the last quarter
    gdp_df['change'] = gdp_df['gdp_chained'].diff()
    #calculate the start point of recessions
    gdp_df['start'] = (gdp_df['change'] < 0) & (gdp_df['change'].shift(-1) < 0) & (gdp_df['change'].shift(1) > 0)
    #calculate the end point of recessions
    gdp_df['end'] = (gdp_df['change'] > 0) & (gdp_df['change'].shift(1) > 0) & (gdp_df['change'].shift(2) < 0)
    #get quarters of starts and ends
    start_qs = gdp_df.index[gdp_df.start]
    maybe_end_qs = gdp_df.index[gdp_df.end]
    end_qs = []
    for i in start_qs:
        end_qs.append(next((j for j in maybe_end_qs if j > i), None))
    #replace end column with pruned end list
    gdp_df['end'] = gdp_df.index == end_qs
    #create column showing when recession occur
    gdp_df['recession'] = False
    for start, end in zip(start_qs, end_qs):
        gdp_df.recession[start:end] = True
    #calculate recession bottom
    bottom_val = gdp_df.gdp_chained[gdp_df.recession].min()
    bottom_q = gdp_df.index[gdp_df.gdp_chained == bottom_val]
    gdp_df.recession[bottom_q] = 'bottom'

    return gdp_df
