import pathlib
import os 
import sys
import itertools
import prediction_model 

PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

DATAPATH = os.path.join(PACKAGE_ROOT,"data_source")

DATA_FILE = 'data_month_full_T52024.csv'

MODEL_NAME = 'best_param.pkl'
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT,'trained_models')

VARIABLE_TO_CONVERT_TYPES = ['based_month']
VARIABLE_FOR_DIFFERENCING = 'gmv'

# Define the parameter grid for SARIMAX model and periods for prediction
p = range(1, 4)  
d = range(0, 2)  
q = range(1, 4)  
P = range(1, 4)  
D = range(0, 2)  
Q = range(1, 4)  
s = [6, 9, 12]

PARAM_GRID = list(itertools.product(itertools.product(p, d, q), itertools.product(P, D, Q, s)))

PERIODS = [1, 3, 6, 12]

print(SAVE_MODEL_PATH)