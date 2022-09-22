import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

vs.TSMB_Decomposed_Multi()
vs.TSMB_Decomposed_Add()
exit()



grp_saleCnt_byItem = df_train.groupby(['item_id'])['item_cnt_day'].sum()
# print(grp_saleCnt_byItem)


"""
df_train columns:
['date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day']
"""


# print(pp.cols_train)
# grp = pp.groupby_df(pp.df_trainPrcd[['date','date_block_num', 'item_cnt_day']], 'date', print_cols=True).sum()
# plt.plot(grp['date_block_num'], grp['item_cnt_day'])
# plt.plot(grp['date'], grp['item_cnt_day'])
# plt.show()
# print(1)