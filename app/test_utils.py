import unittest
import pandas as pd
from utils import *

testfile = 'app/test_data/test_data.csv'

class TestUtils(unittest.TestCase):
    
    def test_load_data(self):
        df = load_data(testfile)
        self.assertEqual(type(df), pd.DataFrame)
        self.assertEqual(df.shape, (9,8))
    
    def test_normalise_data(self):
        df1 = load_data(testfile)
        df2 = normalise_data(df1.copy())
        for col in ['temperature','var1','pressure','windspeed']:
            self.assertTrue(len(np.setdiff1d(df1[col].values, df2[col].values)) >= 1)
            self.assertNotIn(df1[col].values, df2[col].values)
    
    def test_split_attrs_labels(self):
        df = load_data(testfile)
        df = normalise_data(df)
        X, Y = split_attrs_labels(df)
        
        self.assertEqual(X.shape, (9, 6))
        self.assertEqual(Y.shape, (9,1))
        self.assertNotIn(Y, X)

        
if __name__ == '__main__':
    unittest.main()