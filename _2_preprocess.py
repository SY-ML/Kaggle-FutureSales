import pandas as pd

from _1_dataset import DataSet
import datetime

# TIME SERIES
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

class PreProcess(DataSet):
    def __init__(self):
        super().__init__()
        self.df_trainPrcd = self.preprocess_df_train()

    def transform_str2date(self, df, col, format="%d.%m.%Y"):
        # df[col] = df[col].apply(lambda x: datetime.datetime.strptime(x, format))
        df[col] = pd.to_datetime(df[col].astype(str), format=format)

        return df

    def preprocess_df_train(self):
        df = self.df_train
        # df = self.transform_str2date(df)

        return df

