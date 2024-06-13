# model_trainer.py

import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    BaggingRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from source.exception import CustomException
from source.logger import logging
from source.utils import save_object, model_training

# Configuration class for model trainer paths
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

# Model Trainer class responsible for training and evaluating models
class ModelTrainer:
    def __init__(self):
        # Initialize the model trainer configuration
        self.model_trainer_config = ModelTrainerConfig()

    # Method to initiate model training
    def initiate_model_training(self, x_train_array, y_train_array, x_test_array, y_test_array):
        try:
            # Define the training and testing arrays
            self.x_train_array = x_train_array
            self.y_train_array = y_train_array
            self.x_test_array = x_test_array
            self.y_test_array = y_test_array
            logging.info("Defined the training and testing arrays for x and y")

            # Define models and their hyperparameters
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "SVR": SVR(),
                "XGBRegressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=0),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Extra Trees Regressor": ExtraTreesRegressor(),
                "Bagging Regressor": BaggingRegressor()
            }
            params = {
                "Decision Tree": {
                    #'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    #'splitter': ['best', 'random'],
                    'max_depth': [None, 10, 20, 30, 40, 50],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 5]
                },
                "Random Forest": {
                    'n_estimators': [40, 50, 75, 90, 100, 125, 140, 150, 180, 200 , 235, 256],
                    #'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    #'max_features': ['auto', 'sqrt', 'log2'],
                    'max_depth': [None, 10, 20, 30, 40, 50],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 5]
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 4, 5, 6, 7, 8],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 5]
                },
                "Linear Regression": {},
                "SVR": {
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto']
                },
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 4, 5, 6, 7, 8],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
                },
                "CatBoost Regressor": {
                    'iterations': [100, 200, 300],
                    'depth': [4, 6, 8, 10],
                    'learning_rate': [.1, .01, .05, .001]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    'n_estimators': [50, 100, 200]
                },
                "Extra Trees Regressor": {
                    'n_estimators': [50, 100, 200],
                    #'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    #'max_features': ['auto', 'sqrt', 'log2'],
                    'max_depth': [None, 10, 20, 30, 40, 50],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 5]
                },
                "Bagging Regressor": {
                    'n_estimators': [10, 20, 30, 40, 50],
                    'max_samples': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    'max_features': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                }
            }

            # Train models and get their performance metrics
            model_report: dict = model_training(
                param=params,
                models=models,
                x_train_array=x_train_array,
                y_train_array=y_train_array,
                x_test_array=x_test_array,
                y_test_array=y_test_array
            )

            # Get the best model based on performance metrics
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # Raise an exception if no model meets the performance threshold
            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            # Save the best model to the specified path
            save_object(
                obj=best_model,
                file_path=self.model_trainer_config.trained_model_file_path,
            )

            # Predict on the test set and compute the R2 score
            predicted = best_model.predict(x_test_array)
            r2_square = r2_score(y_test_array, predicted)
            return r2_square

        except Exception as e:
            # Raise a custom exception with the error and system details
            raise CustomException(e, sys)
