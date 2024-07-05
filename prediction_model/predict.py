import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pathlib import Path
import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=Warning)

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config  
from prediction_model.processing.data_handling import load_pipeline, load_dataset

load_best_param = load_pipeline()
customer_data = load_dataset(config.DATA_FILE)

def checking_stationary(customer_data, diff):
    best_diff = None
    min_pvalue = float('inf')
    for i in range(diff, diff + 3):
        diff_df = customer_data[config.VARIABLE_FOR_DIFFERENCING] - customer_data[config.VARIABLE_FOR_DIFFERENCING].shift(i)
        res = adfuller(diff_df.dropna())
        p_value = res[1]
        if p_value < min_pvalue:
            min_pvalue = p_value
            best_diff = i
    return best_diff, min_pvalue

def generate_credit_limit(retailer_id, best_params, customer_data):
    customer_data = customer_data[customer_data['retailer_id'] == retailer_id]
    sum_predictions = {} # Save predicted output of 1, 3, 6, 12 months
    for period, param in best_params.items():
        # Making data to be stationary
        diff = checking_stationary(customer_data, period)[0]
        shift_df = customer_data[config.VARIABLE_FOR_DIFFERENCING] - customer_data[config.VARIABLE_FOR_DIFFERENCING].shift(diff)
        shift_df = shift_df.dropna()

        try:
            # define order and seasonal order
            order, seasonal_order = param
            #Run model
            model = SARIMAX(shift_df, order=order, seasonal_order=seasonal_order, enforce_stationarity=False)
            results = model.fit(disp=False)
            prediction = results.forecast(steps=period) # thực hiện predict revenue in next X month
            forecast_origin = prediction + pd.Series(customer_data[config.VARIABLE_FOR_DIFFERENCING][-diff:][:period].values, index = prediction.index)  # convert value back to original value
            sum_predictions[f'forecast_gmv_{period}_month'] = forecast_origin.apply(lambda x: 0 if x < 0 else x).sum() # Nếu kết quả predict ra revenue âm thì replace = 0
        except TypeError as error:
            print(f"The revenue for retailer_id: {retailer_id} cannot be predicted for the next {period} months.")
            continue

    #Create Dataframe contains predicted output of 1, 3, 6, 12 months
    results_df = pd.DataFrame(sum_predictions.items()).T 
    results_df.columns = results_df.iloc[0]
    results_df = results_df[1:]
    results_df['retailer_id'] = retailer_id

    return results_df

if __name__=="__main__":
    forecast_rev = pd.DataFrame()
    for retailer_id, best_param in load_best_param.items():    
        try:
            ret_df = generate_credit_limit(retailer_id, best_param, customer_data)
            forecast_rev = pd.concat([forecast_rev, ret_df], ignore_index = False)
            print("Finish generating credit limit for retailer_id {}".format(retailer_id))
        except ValueError as error:
            print("ValueError occurred for retailer_id {}: {}".format(retailer_id, error))
            continue
    print(forecast_rev)