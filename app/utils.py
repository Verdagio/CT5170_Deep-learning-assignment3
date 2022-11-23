import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sklearn

def load_data(filepath):
    """Loads in data from the filepath of a given csv
    
    Args:
        filepath (string): A string of the file location
    
    Returns:
        dataframe (Pandas.dataFrame) 
    """
    return pd.read_csv(filepath)

def normalise_data(df):
    """Normalises data in the dataframe using a min-max feature scaling and Ordinal encoding

    Args:
        df (Pandas.dataFrame): the dataFrame to be normalised
        
    Returns:
        dataFrame (Pandas.dataFrame): Normalised
    """
    for col in ['temperature','var1','pressure','windspeed']:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    
    enc = LabelEncoder()
    df['var2'] = enc.fit_transform(df['var2'].values)
    return df

def split_attrs_labels(df):
    """Processes the dataFrame and returns data and labels in npArrays

    Args:
        df (pandas.dataframe): Input dataframe after normalisation

    Returns:
        X, Y: returns result of sklearn.model_selection.train_test_split
    """
    # The shape of our data: 
    # ID,datetime,temperature,var1,pressure,windspeed,var2,electricity_consumption
    # ID is not really required here we'll drop that, split the data from datetime to var2
    # if elec_consumption exists we'll return it as y otherwise y is an empty array
    
    X = df.iloc[:,1:7].to_numpy()
    Y = df.iloc[:,7:].to_numpy()

    return X, Y
    
    
if __name__ == '__main__':
    df = load_data('./data/test.csv')
    df = normalise_data(df.iloc[0:25,:])
    X, Y = split_attrs_labels(df)

    print(X, Y)