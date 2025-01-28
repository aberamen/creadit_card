import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataPreprocessor:
    def __init__(self):
        self.numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        self.categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        self.preprocessor = None

    def create_pipeline(self, numerical_cols, categorical_cols):
        """
        Create a preprocessing pipeline for numerical and categorical columns.
        """
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numerical_transformer, numerical_cols),
                ('cat', self.categorical_transformer, categorical_cols)
            ]
        )
    
    def fit_transform(self, X_train):
        """
        Fit and transform the training data.
        """
        if self.preprocessor is None:
            raise ValueError("Pipeline is not initialized. Call `create_pipeline` first.")
        return self.preprocessor.fit_transform(X_train)

    def transform(self, X):
        """
        Transform the data using the fitted preprocessor.
        """
        if self.preprocessor is None:
            raise ValueError("Pipeline is not initialized. Call `create_pipeline` first.")
        return self.preprocessor.transform(X)

# Example usage:
# preprocessor = DataPreprocessor()
# preprocessor.create_pipeline(numerical_cols=['age', 'income'], categorical_cols=['gender', 'state'])
# X_train_transformed = preprocessor.fit_transform(X_train)
# X_test_transformed = preprocessor.transform(X_test)
