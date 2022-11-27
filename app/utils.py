import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import glob
import os

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

    cols = ['ID','datetime','temperature','var1','pressure','windspeed','A','B','C']
    if 'electricity_consumption' in df.columns:
        cols.append('electricity_consumption')

    return df[cols]

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
    X = df.iloc[:,2:9].to_numpy().astype(np.float32)
    Y = df.iloc[:,9:].to_numpy().astype(np.float32)
    return X, Y

# We definitely don't want to generate this every time so this is a work around to save 15 - 25 minutes at a time...
def get_file_map():
    return dict({'2013': {'07': './models/2013-07.h5', '08': './models/2013-08.h5', '09': './models/2013-09.h5', '10': './models/2013-10.h5', '11': './models/2013-11.h5', '12': './models/2013-12.h5'}, '2014': {'07': './models/2014-07.h5', '08': './models/2014-08.h5', '09': './models/2014-09.h5', '10': './models/2014-10.h5', '11': './models/2014-11.h5', '12': './models/2014-12.h5', '01': './models/2014-01.h5', '02': './models/2014-02.h5', '03': './models/2014-03.h5', '04': './models/2014-04.h5', '05': './models/2014-05.h5', '06': './models/2014-06.h5'}, '2015': {'07': './models/2015-07.h5', '08': './models/2015-08.h5', '09': './models/2015-09.h5', '10': './models/2015-10.h5', '11': './models/2015-11.h5', '12': './models/2015-12.h5', '01': './models/2015-01.h5', '02': './models/2015-02.h5', '03': './models/2015-03.h5', '04': './models/2015-04.h5', '05': './models/2015-05.h5', '06': './models/2015-06.h5'}, '2016': {'07': './models/2016-07.h5', '08': './models/2016-08.h5', '09': './models/2016-09.h5', '10': './models/2016-10.h5', '11': './models/2016-11.h5', '12': './models/2016-12.h5', '01': './models/2016-01.h5', '02': './models/2016-02.h5', '03': './models/2016-03.h5', '04': './models/2016-04.h5', '05': './models/2016-05.h5', '06': './models/2016-06.h5'}, '2017': {'01': './models/2017-01.h5', '02': './models/2017-02.h5', '03': './models/2017-03.h5', '04': './models/2017-04.h5', '05': './models/2017-05.h5', '06': './models/2017-06.h5'}})
    
    
if __name__ == '__main__':
    df = load_data('./data/train.csv')
    df = normalise_data(df)
    df.to_csv('./blah.csv')
    batches = get_time_series_batches(df)

    X, Y = split_attrs_labels(batches['2015'].get('02'))
    # print(X, Y.shape)
    fm = get_file_map('./app/models')
    print(fm)

   