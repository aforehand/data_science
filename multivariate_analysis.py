import seaborn as sns
import pandas as pd
import numpy as np


def get_heatmap(df, target_name, k=10):
    corrmat = df.corr()
    in_order = corrmat[target].abs().sort_values(ascending=False)
    cols = in_order.nlargest(k).index
    cm = np.corrcoef(corrmat[cols].values.T)
    sns.set(font_scale=1)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()
