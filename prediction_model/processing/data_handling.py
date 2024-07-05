import os
import pandas as pd
import joblib
import pickle
from pathlib import Path
import sys
import warnings
warnings.simplefilter(action='ignore', category=Warning)

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config

#Load the dataset
def load_dataset(file_name):
    filepath = os.path.join(config.DATAPATH,file_name)
    _data = pd.read_csv(filepath)
    return _data

def save_pipeline(pipeline_to_save):
    save_path = os.path.join(config.SAVE_MODEL_PATH,config.MODEL_NAME)
    joblib.dump(pipeline_to_save, save_path)
    print(f"Model has been saved under the name {config.MODEL_NAME}")

def load_pipeline():
    save_param_path = os.path.join(config.SAVE_MODEL_PATH,config.MODEL_NAME)
    param_loaded = joblib.load(save_param_path)
    print(f"Best param has been loaded")
    return param_loaded

