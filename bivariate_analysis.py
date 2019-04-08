import pandas as pd
import seaborn as sns
from univariate_analysis import *
from sklearn.linear_model import LinearRegression
from scipy import stats

# returns a key/value dataframe of the most influential words
# in a given column
def get_important_words(column, target, num=1000):
    column = clean_strings(column)
    kv_df = get_kv_df(column=column)
    word_features = pd.DataFrame()
    # chooses the most common words to check
    for word in kv_df.key[:min(num, kv_df.key.size)]:
        word_features[word] = column.apply(lambda x: word in x)
    linreg = LinearRegression()
    linreg.fit(word_features, target)
    importance_df = pd.DataFrame(data = {'coef': linreg.coef_, 'word': word_features.columns})
    importance_df.sort_values(by='coef', ascending=False, inplace=True)
    mean = importance_df.coef.mean()
    std = importance_df.coef.std()
    # filters by words with the largest coefficients
    most_important = importance_df[abs(importance_df.coef) > mean+2*std]
    important_kv_df = kv_df[kv_df.key.isin(most_important.word)]
    return important_kv_df


def bool_plots(df, target):
    bool_cols = df.select_dtypes(include=[np.bool]).columns.tolist()
    for i in range(len(bool_cols)):
        sns.boxplot(df[bool_cols[i]], target)
        plt.show()

def cat_plots(df, target):
    cat_cols = df.select_dtypes(include=[np.object]).columns.tolist()
    for i in range(len(cat_cols)):
        sns.violinplot(df[cat_cols[i]], target)
        plt.show()
