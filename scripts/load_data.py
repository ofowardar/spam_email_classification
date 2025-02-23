import pandas as pd
import numpy as np

def load_data(data_path):
    df = pd.read_csv(data_path)
    return df
