import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.tsa.api as smt
"""
REFERENCE
https://www.kaggle.com/code/jagangupta/time-series-basics-exploring-traditional-ts
https://www.kaggle.com/code/dlarionov/feature-engineering-xgboost
https://www.kaggle.com/code/nareshbhat/outlier-the-silent-killer
"""

from _1_dataset import DataSet
from _2_preprocess import PreProcess
from _3_visualize import Visualize
ds = DataSet()
pp = PreProcess()
vs = Visualize()

### Glance at files
# ds.print_data_descriptions()
# ds.show_data_details()
"""
SUMMARY: GLANCE AT FILES
- File Structure
    >> sales_train: date, date_block_num, shop_id, item_[id, price, cnt(day)]
    >> sales_test: ID, shop_id, item_id
    >> submissions: ID, item_cnt(month)
    >> items: item_[name, id, cat]
    >> cat: item_cat_[name, id]
    >> shop: shop_[name, id]
- Data Structure
    >> date - date_block_num
    >> item_id - item_name, cat, price, cnt(day), cnt(month)
    >> item_cat_id - item_cat_name
    >> shop_id - shop_name
- No missing values in all the files.
- Date: object type. #ToDo-Transform

"""

### Glance at columns
# pp.show_column_details()
"""
SUMMARY: GLANCE AT COLUMNS

col[date_block_num].nunique : 34
col[shop_id].nunique : 60
col[item_id].nunique : 22170
col[item_category_id].nunique : 84
"""

# VISUALIZATION: Outliers
df_train = pp.df_trainPrcd

# MONTHLY_SALES ## REF1
# monthly_sales = df_train.groupby(['date_block_num', 'shop_id', 'item_id'])['date', 'item_price', 'item_cnt_day'].agg({'date':['min', 'max'], 'item_price':'mean', 'item_cnt_day':'sum'})
# print(monthly_sales)

### Number of items per cat
# vs.number_of_items_per_category()

### Total Sales Count by month
# vs.total_sales_count_by_month()

# vs.TSMB_Decomposed_Multi()
# vs.TSMB_Decomposed_Add()

# vs.test_stationarity(vs.TSBM)
# vs.TSMB_OrgVsDetrendedVsDeseasonalized()
# vs.test_stationarity(vs.TSBM_DTRD)
# vs.test_stationarity(vs.TSBM_DESND, "TSBM_De-Trended")
"""
Now after the transformations, our p-value for the DF test is well within 5 %. Hence we can assume Stationarity of the series
We can easily get back the original series using the inverse transform function that we have defined above.

Now let's dive into making the forecasts!
"""

### AR, MA, and ARMA models:
# Simulate an AR(1) process with alpha = 0.6
np.random.seed(1)
n_samples = int(1000)
a = 0.6
x = w = np.random.normal(size=n_samples)

for t in range(n_samples):
    x[t] = a*x[t-1] + w[t]
limit=12
# _ = vs.TSBM_plot(x, lags = limit, title="AR(1) process")

# Simulate an AR(2) process
n = int(1000)
alphas = np.array([.444, .333])
betas = np.array([0.])

print(alphas)
print(betas)

# Python requires us to specify the zero-lag value which is 1
# Also note that the alphas for the AR model must be negated
# We also set the betas for the MA equal to 0 for an AR(p) model
# For more information see the examples at statsmodels.org
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

ar2 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n)
_ = vs.TSBM_plot(ar2, lags=12,title="AR(2) process")


# Simulate an MA(1) process
n = int(1000)
# set the AR(p) alphas equal to 0
alphas = np.array([0.])
betas = np.array([0.8])
# add zero-lag and negate alphas
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]
ma1 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n)
limit=12
_ = vs.TSBM_plot(ma1, lags=limit,title="MA(1) process")


# Simulate MA(2) process with betas 0.6, 0.4
n = int(1000)
alphas = np.array([0.])
betas = np.array([0.6, 0.4])
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

ma3 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n)
_ = vs.TSBM_plot(ma3, lags=12,title="MA(2) process")

# Simulate an ARMA(2, 2) model with alphas=[0.5,-0.25] and betas=[0.5,-0.3]
max_lag = 12

n = int(5000) # lots of samples to help estimates
burn = int(n/10) # number of samples to discard before fit

alphas = np.array([0.8, -0.65])
betas = np.array([0.5, -0.7])
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

arma22 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n, burnin=burn)
_ = vs.TSBM_plot(arma22, lags=max_lag,title="ARMA(2,2) process")
# pick best order by aic
# smallest aic value wins
best_aic = np.inf
best_order = None
best_mdl = None

rng = range(5)
for i in rng:
    for j in rng:
        try:
            tmp_mdl = smt.ARMA(arma22, order=(i, j)).fit(method='mle', trend='nc')
            tmp_aic = tmp_mdl.aic
            if tmp_aic < best_aic:
                best_aic = tmp_aic
                best_order = (i, j)
                best_mdl = tmp_mdl
        except: continue


print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))

"""
df_train columns:
['date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day']
"""
