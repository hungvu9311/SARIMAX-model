import pandas as pd
import numpy as np 
from pathlib import Path
import os
import sys

import warnings
warnings.simplefilter(action='ignore', category=Warning)

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config 
import prediction_model.processing.preprocessing as pp 
from prediction_model.processing.data_handling import load_dataset, save_pipeline
import prediction_model.pipeline as pipe 

def finding_best_param():
    train_data = load_dataset(config.DATA_FILE)
    best_param = pipe.preprocessing_credit_limit.fit_transform(train_data)
    save_pipeline(best_param)

if __name__=='__main__':
    print("Starting for training models")
    finding_best_param()
    print("Finished training the models")
