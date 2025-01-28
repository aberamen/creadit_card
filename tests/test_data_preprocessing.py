import pandas as pd
from src.data_processing.data_preprocessing import DataPreprocessor
import numpy as np

def test_pipeline():
    # Mock data
    data = pd.DataFrame({
        'age': [25, np.nan, 30, 35],
        'income': [50000, 60000, np.nan, 70000],
        'gender': ['M', 'F', np.nan, 'M'],
        'state': ['NY', 'CA', 'TX', np.nan]
    })
    
    numerical_cols = ['age', 'income']
    categorical_cols = ['gender', 'state']

    # Initialize pipeline
    preprocessor = DataPreprocessor()
    preprocessor.create_pipeline(numerical_cols, categorical_cols)
    
    # Test fit_transform
    processed_data = preprocessor.fit_transform(data)
    assert processed_data is not None, "Processed data should not be None"
    assert processed_data.shape[0] == data.shape[0], "Processed data should have the same number of rows"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_pipeline()
