# preprocessor.py
import pandas as pd
import numpy as np

class Preprocessor:
    """ TODO """ 
    def __init__(self, df):
        # Initialize the preprocessor with a DataFrame
        self.df = df

    def replace_string(self):
        self.df.replace('Yes', 1, inplace=True)
        self.df.replace('No', 0, inplace=True)
        self.df.replace('Female', 1, inplace=True)
        self.df.replace('Male', 0, inplace=True)
        self.df.replace('Infected', 1, inplace=True)
        self.df.replace('Non-infected', 0, inplace=True)

    def replace_nan(self):
        for col in self.df.columns:
            if self.df[col].median() in (0, 0.5, 1):
                self.df[col].fillna(self.df[col].median(), inplace=True)
            else:
                self.df[col].fillna(self.df[col].mean(), inplace=True)

    def outlier_handle(self):
        for col in self.df.columns:
            if self.df[col].median() not in (0, 0.5, 1):
                upper_limit = self.df[col].mean() + 2 * self.df[col].std()
                lower_limit = self.df[col].mean() - 2 * self.df[col].std()
                self.df[col] = np.clip(self.df[col], lower_limit, upper_limit)
        

    def standarize(self):
        for col in self.df.columns:
            if self.df[col].median() not in (0, 0.5, 1):
                if self.df[col].std() == 0:
                    self.df[col] = 0
                else:
                    self.df[col] = (self.df[col] - self.df[col].mean()) / self.df[col].std()

    def normalize(self):
        for col in self.df.columns:
            min_val = self.df[col].min()
            max_val = self.df[col].max()
            if(min_val != max_val): self.df[col] = (self.df[col] - min_val) / (max_val - min_val)

    def feature_selection(self, y):
        drop = []
        for col in self.df.columns:
            corr = self.df[col].corr(y.squeeze())
            if abs(corr) < 0.1:
                drop.append(col)
        return drop
    
    def drop_columns(self, drop):
        self.df = self.df.drop(columns=drop)

    def get_result(self):
        return self.df.to_numpy()

    def mice(self):        
        copy_df = self.df.copy()
        for col in copy_df.columns:
            copy_df[col].fillna(copy_df[col].mean(), inplace=True)
        
        for _ in range(10):
            for col in self.df.columns:
                other_cols = [c for c in self.df.columns if c != col]

                observing = copy_df[self.df[col].notnull()]
                missing = copy_df[self.df[col].isnull()]

                if not missing.empty:
                    X = observing[other_cols].values
                    y = observing[col].values
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    X_missing = missing[other_cols].values
                    predicted_values = np.dot(X_missing, beta)
                    copy_df.loc[self.df[col].isnull(), col] = predicted_values

        self.df =  copy_df    
