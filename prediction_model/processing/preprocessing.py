from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product
from sklearn.metrics import mean_squared_error

import pandas as pd 
import numpy as np
import os 
import sys 

import warnings
warnings.simplefilter(action='ignore', category=Warning)

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config

class ConvertDataTypes(BaseEstimator,TransformerMixin):
    def __init__(self, variables_to_dtype=None):
        self.variables_to_dtype = variables_to_dtype

    def fit(self, X):
        return self 
    
    def transform(self, X):
        X = X.copy()
        for col in self.variables_to_dtype:
            X[col] = pd.to_datetime(X[col])
        return X 
    
class RemoveInvalidCustomer(BaseEstimator,TransformerMixin):
    def __init__(self, group_col = 'retailer_id', revenue_col = 'gmv', index_col = 'based_month'):
        self.group_col = group_col
        self.revenue_col = revenue_col
        self.index_col = index_col
        self.retailer_to_remove = None 

    def fit(self, X):
        # Calculating sum of revenue over month for each customers
        group_ret = X.groupby(self.group_col)[self.revenue_col].sum()
        # Identify customer with zero revenue
        self.retailer_to_remove = group_ret[group_ret == 0].index
        return self 
    
    def transform(self, X):
        # Remove customer with zero revenue 
        X = X[~X[self.group_col].isin(self.retailer_to_remove)]
        # Set index for date column
        X = X.set_index([self.index_col])
        return X
    
class FindingBestParam(BaseEstimator,TransformerMixin):
    def __init__(self, param_grid, periods):
        self.param_grid = param_grid
        self.periods = periods 
        self.customer_best_params_ = {}

    def checking_stationary(self, customer_data, diff):
        best_diff = None
        min_pvalue = float('inf')
        for i in range(diff, diff + 3):
            diff_df = customer_data[config.VARIABLE_FOR_DIFFERENCING] - customer_data[config.VARIABLE_FOR_DIFFERENCING].shift(i)
            res = adfuller(diff_df.dropna())
            p_value = res[1]
            if p_value < min_pvalue:
                min_pvalue = p_value
                best_diff = i
        return best_diff
    
    def find_best_params(self, customer_data):
        best_params = {}

        for period in self.periods:
            # Define differencing value
            diff = self.checking_stationary(customer_data, period)
            # Convert data to be stationary
            shift_data = customer_data[config.VARIABLE_FOR_DIFFERENCING] - customer_data[config.VARIABLE_FOR_DIFFERENCING].shift(diff)
            shift_data = shift_data.dropna()

            best_mse = np.inf
            best_param = None

            train_size = len(shift_data) - period
            train, test = shift_data[:train_size], shift_data[train_size:]

            for param in self.param_grid:
                order, seasonal_order = param
                try:
                    model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False)
                    results = model.fit(disp=False)
                    forecast = results.forecast(steps=period)
                    mse = mean_squared_error(test, forecast)
                    if mse < best_mse:
                        best_mse = mse
                        best_param = param
                except:
                    continue
            best_params[period] = best_param
        return best_params

    def fit(self, X):
        return self 
    
    def transform(self, X):
        for customer_id, customer_data in X.groupby('retailer_id'):
            best_params = self.find_best_params(customer_data)
            self.customer_best_params_[customer_id] = best_params
            print(f'Finish finding the best param for {customer_id}')
        customer_best_params = self.customer_best_params_
        return customer_best_params