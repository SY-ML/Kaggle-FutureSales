import pandas as pd

class DataSet():
    def __init__(self):
        self.df_train = pd.read_csv("./competitive-data-science-predict-future-sales/sales_train.csv")
        self.df_test = pd.read_csv("./competitive-data-science-predict-future-sales/test.csv")
        self.df_subm = pd.read_csv("./competitive-data-science-predict-future-sales/sample_submission.csv")
        self.df_items = pd.read_csv("./competitive-data-science-predict-future-sales/items.csv")
        self.df_cats = pd.read_csv("./competitive-data-science-predict-future-sales/item_categories.csv")
        self.df_shops = pd.read_csv("./competitive-data-science-predict-future-sales/shops.csv")
        self.df_all = [self.df_train, self.df_test, self.df_subm, self.df_items, self.df_cats, self.df_shops]

        self.cols_train = self.df_train.columns.tolist()
    def print_data_descriptions(self):
        description = """
        File descriptions
            sales_train.csv - the training set. Daily historical data from January 2013 to October 2015.
            test.csv - the test set. You need to forecast the sales for these shops and products for November 2015.
            sample_submission.csv - a sample submission file in the correct format.
            items.csv - supplemental information about the items/products.
            item_categories.csv  - supplemental information about the items categories.
            shops.csv- supplemental information about the shops.
        Data fields
            ID - an Id that represents a (Shop, Item) tuple within the test set
            shop_id - unique identifier of a shop
            item_id - unique identifier of a product
            item_category_id - unique identifier of item category
            item_cnt_day - number of products sold. You are predicting a monthly amount of this measure
            item_price - current price of an item
            date - date in format dd/mm/yyyy
            date_block_num - a consecutive month number, used for convenience. January 2013 is 0, February 2013 is 1,..., October 2015 is 33
            item_name - name of item
            shop_name - name of shop
            item_category_name - name of item category
        This dataset is permitted to be used for any purpose, including commercial use.
        """
        print(description)
    def setup_printAll (self, col_all = True, row_all = True):
        if col_all is True:
            pd.set_option('display.max_columns', None)
            print("Showing all columns")
        if row_all is True:
            pd.set_option('display.max_rows', None)
            print("Showing all rows")

    def show_data_details(self):
        for i, df in enumerate(self.df_all):
            print(f" === FILE ({i+1}/{len(self.df_all)}) ===")
            print("COLUMNS:\n", df.columns.tolist(),'\n')
            print("HEAD:\n", df.head(), '\n',"TAIL:\n", df.tail(),'\n')
            print("INFO:\n", df.info(), '\n')
            print("DESCRIBE:", df.describe(), '\n')
            print("MISSING VALUES: \n", df.isnull().sum(), '\n')
            print("=====")
            print(f" === FILE ({i+1}/{len(self.df_all)}) ===")
            next = input("WANT TO READ THE NEXT FILE? [Y]/n ")
            if next.lower() == 'y':
                continue
            else:
                break

    def show_column_details(self):
        for i, df in enumerate(self.df_all):
            print(f" === FILE ({i + 1}/{len(self.df_all)}) ===")
            cols = df.columns.tolist()
            print("COLUMNS:\n", df.columns.tolist())
            for col in cols:
                print(f"col[{col}].nunique : {df[col].nunique()}")
                print(f"col[{col}].null.count : {df[col].isnull().sum()}")
                print("-")
            print("=")
            print("COLUMNS:\n", df.columns.tolist(), '\n')
            next = input("WANT TO READ THE NEXT FILE? [Y]/n ")
            if next.lower() == 'y':
                continue
            else:
                break

    def groupby_df(self, df, by_cols, print_cols=False):
        try:
            is_list = len(by_cols)
        except:
            by_cols = list(by_cols)

        if print_cols is True:
            print(df.columns.tolist())

        grp = df.groupby(by_cols, as_index=False)

        return grp




