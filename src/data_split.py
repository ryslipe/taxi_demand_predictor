# split the data into train and test
from datetime import datetime
from typing import Tuple

import pandas as pd

# train test split
def train_test_split(
        df: pd.DataFrame,
        cutoff_date: datetime,
        target_column_name: str
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    '''Create train and test splits'''
    # first split into training and testing dataframes
    train_data = df[df['pickup_hour'] < cutoff_date].reset_index(drop = True)
    test_data = df[df['pickup_hour'] >= cutoff_date].reset_index(drop = True)

    # next create X_train, y_train, X_test, y_test
    X_train = train_data.drop(columns = [target_column_name])
    y_train = train_data[target_column_name]

    X_test = test_data.drop(columns = [target_column_name])
    y_test = test_data[target_column_name]

    return X_train, y_train, X_test, y_test
