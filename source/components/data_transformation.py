# data_transformation.py

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split

from source.exception import CustomException
from source.logger import logging
import os

from source.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join("artifacts", "Preprocessor.pkl")
    logging.info("Defined the preprocessor path.")

class DataTransformation:
    
    def __init__(self):
        self.preprocessor_path = DataTransformationConfig()
    
    def get_preprocessor(self):
        try:
            # Define numeric and categorical features
            num_features = ['votes', 'cost']
            cat_features = ['online_order', 'book_table', 'rest_type', 'type', 'city']
            logging.info("Defined the numeric and categorical features.")

            # Numeric pipeline: Imputation and Scaling
            num_pipeline = Pipeline(steps=[
                ("imputing", SimpleImputer(strategy="mean")),
                ("scaling", StandardScaler())
            ])

            # Categorical pipeline: Imputation and Encoding
            cat_pipeline = Pipeline(steps=[
                ("imputing", SimpleImputer(strategy="most_frequent")),
                ("encoding", OneHotEncoder(handle_unknown='ignore'))
            ])
            logging.info("Defined the numeric and categorical pipelines.")

            # Combine pipelines using ColumnTransformer
            preprocessor = ColumnTransformer([
                ("num_features", num_pipeline, num_features),
                ("cat_features", cat_pipeline, cat_features)
            ])
            logging.info("Defined the complete preprocessor.")
            return preprocessor

        except Exception as e:
            logging.error(f"Error in get_preprocessor: {e}")
            raise CustomException(e)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            # Load datasets
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)

            target = "rate"
            y_train = train_data[target]
            x_train = train_data.drop(target, axis=1)
            x_test = test_data.drop(target, axis=1)
            y_test = test_data[target]

            logging.info("Split data into features and target.")
            
            # Initialize preprocessor
            preprocessor = self.get_preprocessor()
            logging.info("Fetched the preprocessor successfully.")

            # Fit the preprocessor on training data
            preprocessor.fit(x_train)
            save_object(preprocessor, self.preprocessor_path.preprocessor_path)
            logging.info("Saved the preprocessor object successfully.")

            # Transform training and testing data
            x_train_array = preprocessor.transform(x_train).toarray()
            y_train_array = np.array(y_train)
            x_test_array = preprocessor.transform(x_test).toarray()
            y_test_array = np.array(y_test)

            logging.info(f"Data transformation complete: "
                         f"x_train_array shape: {x_train_array.shape}, "
                         f"y_train_array shape: {y_train_array.shape}, "
                         f"x_test_array shape: {x_test_array.shape}, "
                         f"y_test_array shape: {y_test_array.shape}")

            return x_train_array, y_train_array, x_test_array, y_test_array

        except Exception as e:
            logging.error(f"Error in initiate_data_transformation: {e}")
            raise CustomException(e, sys)

# Example usage
# data_transformation_obj = DataTransformation()
# x_train_array, y_train_array, x_test_array, y_test_array = data_transformation_obj.initiate_data_transformation(train_data_path='path/to/train.csv', test_data_path='path/to/test.csv')
