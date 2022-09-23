import pandas as pd

from _1_dataset import DataSet
from _2_preprocess import PreProcess

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf, arma_order_select_ic

class Visualize(PreProcess):
    def __init__(self):
        super().__init__()
        self.TSBM = self.TOTALSALESCOUNT_BYMONTH(show_plot=False)
        _, self.TSBM_DTRD, self.TSBM_DESND = self.TSMB_OrgVsDetrendedVsDeseasonalized(show_plots=False)

    """
    
    """

    def showTitleLabels(self, title=None, x_label=None, y_label=None, show=True):
        if title is not None:
            plt.title(title)
        if x_label is not None:
            plt.xlabel(x_label)
        if y_label is not None:
            plt.ylabel(y_label)

        if show is True:
            plt.tight_layout()
            plt.show()

    def axTitleLabels(self, ax, title=None, x_label=None, y_label=None, show=True):
        if title is not None:
            ax.set_title(title)
        if x_label is not None:
            ax.set_xlabel(x_label)
        if y_label is not None:
            ax.set_ylabel(y_label)
        if show is True:
            plt.tight_layout()
            plt.show()

    def number_of_items_per_category(self, show_plot=True):
        df_items = self.df_items
        grp_itemCnt_byCat = df_items.groupby(['item_category_id'], as_index=False).count()
        group_byCat = grp_itemCnt_byCat.sort_values(by='item_id')
        if show_plot is True:
            sns.barplot(group_byCat['item_category_id'], group_byCat['item_id'])
            self.showTitleLabels("ITEM_ID COUNT BY CATEGORY ID", 'item_category_id', 'item_id count')
        return grp_itemCnt_byCat


    def TOTALSALESCOUNT_BYMONTH(self, show_plot=True):
        df_train = self.df_train
        grp_totalSales_byMonth = df_train.groupby(['date_block_num'])['item_cnt_day'].sum()
        if show_plot is True:
            plt.plot(grp_totalSales_byMonth)
            self.showTitleLabels("ITEM SALES COUNT BY MONTH", 'MONTH', 'COUNT')

        return grp_totalSales_byMonth

    def TSBM_RollingMeanStd(self):
        grp_totalSales_byMonth = self.TSBM
        rolling_mean = grp_totalSales_byMonth.rolling(window=12, center=False).mean()
        rolling_std = grp_totalSales_byMonth.rolling(window=12, center=False).std()

        plt.plot(rolling_mean, label='Rolling Mean')
        plt.plot(rolling_std, label='Rolling Std')
        self.showTitleLabels("ROLLING MEAN / STD by MONTH", "MONTH", "COUNT")
        """
        There is an obvious "seasonality" (Eg: peak sales around a time of year) and a decreasing "Trend".
        """
        return rolling_mean, rolling_std

    def TSMB_Decomposed_Multi(self):
        df = self.TSBM
        decomp = seasonal_decompose(df.values, period =12, model='multiplicative')

        # decomp.observed.plot()
        # decomp.trend.plot()
        # decomp.seasonal.plot()
        # decomp.resid.plot()
        decomp.plot()
        self.showTitleLabels("DECOMPOSED TOTAL SALES COUNT BY MONTH", 'MONTH')

    def TSMB_Decomposed_Add(self):
        df = self.TSBM
        decomp = sm.tsa.seasonal_decompose(df.values, period=12, model='multiplicative')
        decomp.plot()
        self.showTitleLabels("DECOMPOSED TOTAL SALES COUNT BY MONTH", 'MONTH')

    def test_stationarity(self, timeseries, dataName = ""):
        # Perform Dickey-Fuller test
        message = f"\n== ADF TEST RESULT ==" if dataName =="" else f"\n== [{dataName}] ADF TEST RESULT =="

        print(message)
        df_test = adfuller(timeseries, autolag="AIC")
        df_output = pd.Series(df_test[0:4], index=['Test Statistic', 'p-value', '#Lags Used','Number of Observations Used'])

        for key, value in df_test[4].items():
            df_output['Critical Value (%s)'%key] = value
        print(df_output)

    def create_difference(self, dataset, interval = 1):
        diff = []
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i-interval]
            diff.append(value)
        return pd.Series(diff)

    def invert_difference(self, last_ob, value):
        return value + last_ob


    def TSMB_OrgVsDetrendedVsDeseasonalized(self, show_plots = True):
        df = self.TSBM # original data
        df_dt = self.create_difference(df) # de-trended data. next month - current month
        df_ds = self.create_difference(df, 12) # de-seasonalized data, # month of the next year - month of this year

        if show_plots is True:
            fig, ax = plt.subplots(3, 1, figsize=(18, 10))
            ax[0].plot(df)
            self.axTitleLabels(ax[0], "ORIGINAL DATA", 'TIME', 'ITEMCNT/DAY', show=False)
            ax[1].plot(df_dt)
            self.axTitleLabels(ax[1], "DE-TRENDED DATA", 'TIME', 'ITEMCNT/DAY', show=False)
            ax[2].plot(df_ds)
            self.axTitleLabels(ax[2],"DE-SEASONALIZED DATA", 'TIME', 'ITEMCNT/DAY', show=True)

        return df, df_dt, df_ds

