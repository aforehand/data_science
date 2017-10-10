import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from pandas import DataFrame

readRDS = robjects.r['readRDS']
summary_df = readRDS('exdata_data_NEI_data/summarySCC_PM25.rds')
summary_df = pandas2ri.ri2py(summary_df)
summary_df.to_csv('exdata_data_NEI_data/summarySCC_PM25.csv')
scc_df = readRDS('exdata_data_NEI_data/Source_Classification_Code.rds')
scc_df = pandas2ri.ri2py(scc_df)
scc_df.to_csv('exdata_data_NEI_data/Source_Classification_Code.csv')
