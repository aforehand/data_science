import pandas as pd



# returns a list of features with significantly different
# target value distributions
def select_bool_features(df, target, bool_cols=None, p=0.01):
    pvalues=[]
    if bool_cols is None:
        bool_cols = df.select_dtypes(include=[np.bool]).columns.tolist()
    for col in bool_cols:
        group_a = target[df[col]==True]
        group_b = target[df[col]==False]
        pvalues.append(stats.ttest_ind(group_a, group_b, equal_var=False).pvalue)
    significance_df = pd.DataFrame({'feature': bool_cols, 'pvalue': pvalues})
    significance_df = significance_df[significance_df.pvalue<=p]
    return significance_df.feature.tolist()

def select_numeric_features(df, target, num_cols=None, r=0.3):
    correlations = []
    if num_cols is None:
        num_cols = df.select_dtypes(include=[np.float, np.int]).columns.tolist()
    for col in num_cols:
        correlations.append(pearsonr(df[col], target)[0])
    corr_df = pd.DataFrame({'feature': num_cols, 'correlation': correlations})
    corr_df.sort_values(by='correlation', ascending=False, inplace=True)
    corr_df.reset_index(drop=True, inplace=True)
    corr_df = corr_df[abs(corr_df.correlation) >=r]
    return corr_df.feature.tolist()
