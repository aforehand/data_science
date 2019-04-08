import pandas as pd
import numpy as np

pd.options.display.float_format = '{:.2f}'.format

def get_nulls(df):
    total = train.isnull().sum().sort_values(ascending=False)
    percent = (train.isnull().sum()*100/train.isnull().count()).sort_values(ascending=False)
    missing = pd.concat([total,percent],axis=1,keys=['total','percent'])
    print(f'{missing}\n')

def get_num_cat_cols(df):
    num_cols = [col for col in df.columns if np.issubdtype(df[col].dtype, np.number)]
    cat_cols = [col for col in df.columns if np.issubdtype(df[col].dtype, np.object)]
    return num_cols, cat_cols

def get_info(df):
    print('SHAPE')
    print(f'rows: {df.shape[0]}\tcolumns: {df.shape[1]}\n')
    print('DTYPES')
    print(f'{df.info(null_counts=None)}\n')
    print('STATS')
    print(df.describe().T)
    print(df.describe(include=['O']))
    print(f'\n')
    print('NULLS')
    get_nulls(df)
