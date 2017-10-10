from pandas import DataFrame, Series
import pandas as pd
import numpy as np

#turns a nested bulleted list into a nice clean CSV
def clean_towns():
    #read text file into DataFrame
    raw_df = pd.read_table('college_towns.txt', sep='\n', header=None)
    #partition to get states into their own column
    part_df = raw_df[0].str.partition('\t')
    #forward fill states so each town has its state in its row
    part_df = part_df.replace('', np.nan)
    part_df[0] = part_df[0].ffill()
    #drop na rows
    part_df = part_df.dropna()

    #partition to get neighborhoods into their own column
    part_df[[2,3,4]] = part_df[2].str.partition('◦\t', expand=True)
    #remove delimiter rows
    part_df = part_df.drop([1,3], axis=1)
    #rename columns
    part_df.columns = ['state', 'town', 'neighborhood']
    #remove remaining bullets
    towns_clean = part_df.replace('^•\t', '', regex=True)
    #remove text after town name
    towns_clean = towns_clean.replace('(\[| \(|\u00A0).*$', '', regex=True)

    #forward fill cities that have neighborhoods
    towns_filled = towns_clean.replace('', np.nan)
    towns_filled['town'] = towns_filled['town'].ffill()
    #save DataFrame to CSV
    towns_filled.to_csv('college_towns.csv')

    # #for turning a column into headers
    # #create a dict with states as keys and town lists as values
    # towns_dict = {k: list(v) for k,v in towns_clean.groupby(0)[2]}
    # #create DataFrame from dict
    # towns_df = DataFrame(dict([(k,Series(v)) for k,v in towns_dict.items()]))
