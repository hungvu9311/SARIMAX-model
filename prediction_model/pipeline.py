import pandas as pd
import numpy as np 
from sklearn.pipeline import Pipeline 
from pathlib import Path
import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=Warning)

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config 
# import prediction_model.processing.preprocessing as pp 
from prediction_model.processing.preprocessing import ConvertDataTypes, RemoveInvalidCustomer, FindingBestParam
from prediction_model.processing.data_handling import load_dataset, save_pipeline

preprocessing_credit_limit = Pipeline(
    [
        ('ConvertDataTypes', ConvertDataTypes(variables_to_dtype=config.VARIABLE_TO_CONVERT_TYPES)),
        ('RemoveInvalidCustomer', RemoveInvalidCustomer()),
        ('FindingBestParam', FindingBestParam(param_grid=config.PARAM_GRID, periods=config.PERIODS))
    ]
)
