import numpy as np
import pandas as pd

def load_data(filepath):
    """Loads in data from the filepath of a given csv
    
    Args:
        filepath (string): A string of the file location
    
    Returns:
        dataframe (Pandas.dataFrame) 
    """
    return pd.read_csv(filepath)

def normalise_data(df):
    """Normalises data in the dataframe using a min-max feature scaling

    Args:
        df (Pandas.dataFrame): the dataFrame to be normalised
        
    Returns:
        dataFrame (Pandas.dataFrame): Normalised
    """
    for col in ['temperature','var1','pressure','windspeed','electricity_consumption']:
        print(type(df[col]))
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df
    
    
if __name__ == '__main__':
    load_data('./data/train.csv')