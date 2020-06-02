# University Rank Predictor

I was curious about how US News determined its university rankings, so I combined the [2017 US News University Rankings](https://github.com/aforehand/data_science/blob/master/education/us_news_university_rankings.csv) with the [2016 Dept. of Education College Scorecard](https://github.com/aforehand/data_science/blob/master/education/college_scorecard.csv) to see which publicly available features were most strongly correlated with rank. I determined that completion rate was by far the best predictor of rank.

After I performed this analysis I found US News' [explanation of ther ranking calculations](https://www.usnews.com/education/best-colleges/articles/how-us-news-calculated-the-rankings). Interestingly, completion rate was said to account for only about 18% of rank.

## Data Processing

To begin, there were a lot of discrepancies in how the university names were represented in each data set. For example, the scorecard left out the name of the main campus of universities with multiple campuses. I fixed these discrepancies in [clean_uni_names.ipynb](https://github.com/aforehand/data_science/blob/master/education/clean_uni_names.ipynb).

I did the rest of the preprocessing in [uni_rankings.ipynb](https://github.com/aforehand/data_science/blob/master/education/uni_rankings.ipynb). I began by merging the two datasets, removing duplicate columns, addressing nulls, and fixing incorrect dtypes.

The scorecard had 122 columns, so I looked for columns I could drop. I dropped any column with only one value, columns with two values that were extremely imbalanced, and columns that were highly correlated with others, e.g. quartile info for standardized tests.

## Modeling

I chose Random Forest for my first model so I could avoid encoding all the categorical data and standardizing everything. I chose the hyperparameters with grid search. The model had a train score of .97 and a test score of .94. I looked at the feature importances and found that completion rate was by far the best predictor of rank, having a correlation of -0.94:
![Completion rate vs. rank](https://github.com/aforehand/data_science/blob/master/education/completion%20vs%20rank.png)

I found this very interesting, so I returned to the scorecard dataset to see if there were schools that did not appear on the US News University Rankings but had a high completion rate. Of the 10 schools with the highest completion rate, 7 turned out to be in the top 10 of other US News ranking lists, and 2 were in the top 15. Eight of these were on the National Liberal Arts Colleges list, so completion rate appears to be a good predictor for those schools as well.

