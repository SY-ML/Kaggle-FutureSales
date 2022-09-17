from _1_dataset import DataSet
from _2_preprocess import PreProcess

import matplotlib.pyplot as plt
import seaborn as sns

class Visualize(PreProcess):
    def __init__(self):
        super().__init__()

    def groupby_df(self, df, by_cols, load_col):
        grp = df.groupby(by_cols)[load_col]

        return grp

    # def

