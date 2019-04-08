import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk import stem
from nltk.tokenize import word_tokenize

# returns the number of values in a df row
def how_many(row):
    if type(row) == str:
        try:
            row = eval(row)
        except:
            row = row.split()
    if type(row) == list:
        return len(row)
    else:
        return 1

# returns a column of lowercaes strings with puctuation removed
def clean_strings(column):
    column = column.str.replace('[^\w\s]','')
    column = column.apply(lambda x: x.lower())
    column = column.apply(lambda x: word_tokenize(x))
    lancaster = stem.lancaster.LancasterStemmer()
    column = column.apply(lambda x: [lancaster.stem(word) for word in x])
    return column

# returns a list of values in a given row
def format_row(row):
    if type(row) == str:
        try:
            row = eval(row)
            if not (type(row)==list or type(row)==dict):
                row = str(row).split()
        except:
            row = row.split()
    elif type(row) == dict:
        row = [row]
    return row

# returns a frequency dict for the values in a df column
def get_freq_dict(column, key='name', k=None, v=None):
    freq_dict = {}
    for row in column:
        row = format_row(row)
        if not (row is None or len(row)==0):
            for d in row:
                if type(d) == dict: #if the list elements are dicts
                    # checks if the value depends on another k/v pair
                    if (k is None and v is None) or d[k]==v:
                        if d[key] in freq_dict.keys():
                            freq_dict[d[key]] += 1
                        else:
                            freq_dict[d[key]] = 1
                else: #if the list elements are strings
                    if d in freq_dict.keys():
                        freq_dict[d] += 1
                    else:
                        freq_dict[d] = 1
    return freq_dict

# returns the column frequency of values in a df row
def get_freq_score(row, freq_dict, key='name', value=None, k=None, v=None):
    freq_score = 0
    row = format_row(row)
    if not (row is None or len(row)==0):
        for d in row:
            # checks if the value depends on another k/v pair
            if (k is None and v is None) or d[k]==v:
                if d[key] in freq_dict.keys():
                    if value: # restricts counting to given value
                        if d[key]==value:
                            freq_score += freq_dict[value]
                    else:
                        freq_score += freq_dict[d[key]]
    return freq_score

# returns the number of occurrances of the given values
def get_value_count(row, key='name', value=None, k=None, v=None):
    count = 0
    row = format_row(row)
    if not (row is None or len(row)==0):
        for d in row:
            # checks if the value depends on another k/v pair
            if (k is None and v is None) or d[k]==v:
                if not value is None: # restricts counting to given value
                    if d[key]==value:
                        count += 1
                else:
                    count += 1
    return count

# returns a column with the frequency of a value if that value is in
# each row or the sum of the frequencies of all values if no value
# is given.
# k and v are optional key/value pair that the frequency counting depends on
def get_freq_col(column, freq_dict = None, key='name', value=None, k=None, v=None):
    if freq_dict is None:
        freq_dict = get_freq_dict(column, key, k, v)

    freq_col = column.apply(lambda x: get_freq_score(x, freq_dict, key, value, k, v))
    return freq_col

# returns a column with values extracted from dicts in another column
def get_value_col(row, key='name', k=None, v=None):
    values = []
    row = format_row(row)
    for d in row:
        if (k is None and v is None) or d[k]==v:
            if key in d.keys():
                values.append(d[key])
    return values

# returns a key/value dataframe generated from a frequency dict or
# column. removes stopwords and empty strings and sorts by frequency
def get_kv_df(freq_dict=None, column=None, key='name', k=None, v=None):
    if freq_dict == None:
        freq_dict = get_freq_dict(column, key, k, v)

    df =  pd.DataFrame({'key': [k for k in freq_dict.keys()], 'value': [ v for v in freq_dict.values()]})
    df = df[~df.key.isin(stopwords.words('english'))]
    df = df[df.key!='']
    df.sort_values(by='value', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

#distribution plots for numeric data
def dist_plots(df, bins=None, kde=True):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for i in range(len(num_cols)):
        sns.distplot(train[num_cols[i]])
        plt.figtext(0.7,.8,f'skew: {train[num_cols[i]].skew():.2f}', size='medium')
        plt.figtext(0.7,.75,f'kurt: {train[num_cols[i]].kurt():.2f}', size='medium')
        plt.show()

#bar plots for categorical data
def bar_plots(df):
    cat_cols = df.select_dtypes(include=[np.object]).columns.tolist()
    # fig, axes = plt.subplots(len(cat_cols),1, figsize=(5,5*len(cat_cols)))
    for i in range(len(cat_cols)):
        sns.barplot(df[cat_cols[i]], ax=axes[i])
        plt.show()

# returns the z score of an element in a given column
