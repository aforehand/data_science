import numpy as np
import pandas as pd
from pandas import DataFrame
import scipy.stats as stats

gdp_df = pd.read_csv('quarterly_gdp_2000-2017.csv')
housing_df = pd.read_csv('quarterly_mean_housing_prices.csv')
towns_df = pd.read_csv('college_towns.csv')

def run_ttest():
    college_housing = housing_df.merge(towns_df, on = ['State', 'RegionName', 'Metro'])
    non_college_housing = housing_df[~housing_df.isin(college_housing)]

    start = gdp_df.quarter[gdp_df.start]
    bottom = gdp_df.quarter[gdp_df.recession == 'bottom']

    college_df = DataFrame()
    college_df['start_value'] = college_housing[start]
    college_df['bottom_value'] = college_housing[bottom]
    college_df['difference'] = college_df.bottom_value - college_df.start_value
    college_df['percent_diff'] = college_df.difference / college_df.start_value
    college_df.dropna(inplace=True)
    college_df.to_csv('college_town_recession_housing_prices.csv')

    non_college_df = DataFrame()
    non_college_df['start_value'] = non_college_housing[start]
    non_college_df['bottom_value'] = non_college_housing[bottom]
    non_college_df['difference'] =  non_college_df.bottom_value - non_college_df.start_value
    non_college_df['percent_diff'] = non_college_df.difference / non_college_df.start_value
    non_college_df.dropna(inplace=True)
    non_college_df.to_csv('non_college_town_recession_housing_prices.csv')

    result = stats.ttest_ind(college_df.percent_diff, non_college_df.percent_diff)
    different = (result.pvalue < 0.01)
    better = 'college town' if result.statistic > 0 else 'non-college town'

    return (different, result.pvalue, better)
