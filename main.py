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

## Glance at files
# ds.print_data_descriptions()
ds.show_data_details()

# print(pp.transform_str2date(pp.df_train, 'date'))
# print(pp.df_train.info())
# print(pp.df_train.isnull().sum())

