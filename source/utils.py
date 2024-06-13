# utils.py

import os
import sys
import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from source.logger import logging
from source.exception import CustomException

def save_object(obj, file_path):
    """
    Save an object to a file using dill.
    
    Args:
        obj: The object to be saved.
        file_path: The path to the file where the object will be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        
        # Create directory if it does not exist
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as f:
            logging.info(f"Saving {obj} to {file_path}")
            dill.dump(obj, f)
    
    except Exception as e:
        logging.error(f"Error saving object: {e}")
        raise CustomException(e, sys)

def model_training(param, models, x_train_array, y_train_array, x_test_array, y_test_array):
    """
    Train multiple models using hyperparameter tuning and return performance report.
    
    Args:
        param: Dictionary of hyperparameters for each model.
        models: Dictionary of models to be trained.
        x_train_array: Training data features.
        y_train_array: Training data target.
        x_test_array: Testing data features.
        y_test_array: Testing data target.
    
    Returns:
        report: Dictionary containing R2 score of each model on the test data.
    """
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Training {model_name}")
            para = param[model_name]

            # Perform randomized search for hyperparameter tuning
            rs = RandomizedSearchCV(model, para, cv=3, n_iter=100, random_state=42, n_jobs=-1)
            rs.fit(x_train_array, y_train_array)
            
            # Set best parameters and retrain the model
            model.set_params(**rs.best_params_)
            model.fit(x_train_array, y_train_array)

            # Predict on training and test data
            y_train_pred = model.predict(x_train_array)
            y_test_pred = model.predict(x_test_array)

            # Calculate R2 scores
            train_model_score = r2_score(y_train_array, y_train_pred)
            test_model_score = r2_score(y_test_array, y_test_pred)

            logging.info(f"{model_name} - Train R2: {train_model_score}, Test R2: {test_model_score}")

            report[model_name] = test_model_score

        return report

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load an object from a file using dill.
    
    Args:
        file_path: The path to the file from which the object will be loaded.
    
    Returns:
        The loaded object.
    """
    try:
        with open(file_path, "rb") as file_obj:
            logging.info(f"Loading object from {file_path}")
            return dill.load(file_obj)
    
    except Exception as e:
        logging.error(f"Error loading object: {e}")
        raise CustomException(e, sys)
