import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import json

def load_data(filepath):
    """Loads in data from the filepath of a given csv
    
    Args:
        filepath (string): A string of the file location
    
    Returns:
        dataframe (Pandas.dataFrame) 
    """
    return pd.read_csv(filepath)

def normalise_data(df):
    """Normalises data in the dataframe using a min-max feature scaling and one hot encoding for categorical column

    Args:
        df (Pandas.dataFrame): the dataFrame to be normalised
        
    Returns:
        dataFrame (Pandas.dataFrame): Normalised
    """
    for col in ['temperature','var1','pressure','windspeed']:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    
    # Do some nomalisation on the var2, it's some categorical data but we will add a col for each category deal with it in the future
    enc = LabelEncoder()
    df['var2'] = enc.fit_transform(df['var2'].values)
    var_2 = df.pop('var2')
    one_hot_enc = (lambda x, tgt: (x == tgt)*1.0)

    for l, n in zip(['A', 'B', 'C'], [0,1,2]):
        df[l] = one_hot_enc(var_2, n)

    return df[['ID','datetime','temperature','var1','pressure','windspeed','A','B','C','electricity_consumption']]

def get_time_series_batches(df):
    """create time series batches

    Args:
        df (pandas.Dataframe): input
        
    Return:
        series (dict): a dict containing batches of data by year / mo
        gives an easy utility to be able to grab a batch by year and month e.g. : 
        `time_series_batches['2015'].get('11')`
    """
    m_df = df.copy()

    m_df['year'] = pd.to_datetime(m_df['datetime']).dt.to_period('Y')
    m_df['month'] = pd.to_datetime(m_df['datetime']).dt.to_period('M')

    time_series_batches = dict()
    for year in m_df['year'].unique():
        months = dict()
        for month in m_df['month'].dt.month.unique():
            if month < 10:
                month = f"0{month}"
            mo_data = df[df['datetime'].str.contains(f"{year}-{month}")]
            if not mo_data.empty:
                months[f"{month}"] = mo_data
        time_series_batches[f"{year}"] = months
    
    return time_series_batches

def split_attrs_labels(df):
    """Processes the dataFrame and returns data and labels in npArrays

    Args:
        df (pandas.dataframe): Input dataframe after normalisation

    Returns:
        X, Y: returns result of sklearn.model_selection.train_test_split
    """
    X = df.iloc[:,2:9].to_numpy()
    Y = df.iloc[:,9:].to_numpy()
    return X, Y
    
    
if __name__ == '__main__':
    df = load_data('./data/train.csv')
    df = normalise_data(df)
    df.to_csv('./blah.csv')
    batches = get_time_series_batches(df)

    X, Y = split_attrs_labels(batches['2015'].get('02'))
    # print(X, Y.shape)

   