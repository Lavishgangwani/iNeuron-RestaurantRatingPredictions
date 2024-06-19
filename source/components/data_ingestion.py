# data_ingestion.py

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from source.exception import CustomException
from source.logger import logging
from source.components.data_transformation import DataTransformation
from source.components.model_trainer import ModelTrainer

# Configuration class for data ingestion paths
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

# Data Ingestion class responsible for ingesting and splitting the dataset
class DataIngestion:
    def __init__(self):
        # Initialize the ingestion configuration
        self.ingestion_config = DataIngestionConfig()

    # Method to initiate data ingestion process
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the dataset into a pandas DataFrame
            df = pd.read_csv('notebook/data/Zomato_5k.csv')
            logging.info('Read the dataset as dataframe')

            # Ensure the directory for saving the artifacts exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data to the specified path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved")

            # Split the dataset into training and testing sets
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the training and testing data to the specified paths
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of the data is completed")

            # Return the paths to the training and testing data
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            # Raise a custom exception with the error and system details
            raise CustomException(e, sys)

# Main execution
if __name__ == "__main__":
    # Create a DataIngestion object
    data_ingestion_obj = DataIngestion()
    # Initiate data ingestion and get paths to the train and test data
    train_path, test_path = data_ingestion_obj.initiate_data_ingestion()

    # Create a DataTransformation object
    data_transformation_obj = DataTransformation()
    # Initiate data transformation and get transformed arrays
    x_train_array, y_train_array, x_test_array, y_test_array = data_transformation_obj.initiate_data_transformation(
        train_data_path=train_path, test_data_path=test_path)

    # Create a ModelTrainer object
    model_trainer_obj = ModelTrainer()
    # Initiate model training and print the results
    print(model_trainer_obj.initiate_model_training(
        x_train_array=x_train_array,
        y_train_array=y_train_array,
        x_test_array=x_test_array,
        y_test_array=y_test_array
    ))
