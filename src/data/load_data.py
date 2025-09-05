import pandas as pd
import numpy as np
import os
from typing import Tuple

class DataLoader:
    """Class tp load and validate the dataset"""

    def __init__(self, file_path:str):
        self.file_path = file_path
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """Load the csv data with validation"""
        try:
            self.data = pd.read_csv(self.file_path)
            print(f'Data Loaded Sucessfully. Shape:{self.data.shape}')
            return self.data
        except FileNotFoundError:
            raise FileNotFoundError(f'File Not Found at path: {self.file_path}')
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    
    def validate_data(self) -> dict:
        """Perform Basic Data Validation"""

        if self.data is None:
            raise ValueError('Data not Loaded. Call load_data() first.')

        validation_report ={
            'missing_values': self.data.isnull().sum().to_dict(),
            'duplicate_rows': self.data.duplicated().sum(),
            'data_types': self.data.dtypes.to_dict(),
            'shape': self.data.shape

        }
        return validation_report
    
    def get_feature_target(self, target_column: str = 'BeatsPerMinute') -> Tuple[pd.DataFrame, pd.Series]:
        """Seprate features and target Variable"""
        if target_column not in self.data.columns:
            raise ValueError(f'Target data: {target_column} not found in data')
        
        X = self.data.drop(column = [target_column])
        y = self.data[target_column]

        return X, y


#Example Usage 
if __name__ == "__main__":
    loader = DataLoader('./data/raw/train_less.csv')
    data = loader.load_data()
    validation = loader.validate_data()
    print("Validation Data", validation)



