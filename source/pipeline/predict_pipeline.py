# predict_pipeline.py

from source.logger import logging
from source.exception import CustomException
import numpy as np
import pandas as pd
from source.utils import load_object

class Prediction:
    """
    A class for making predictions using a pre-trained model and preprocessor.
    """

    def __init__(self, online_order, book_table, votes, rest_type, cost, type, city):
        """
        Initialize the Prediction class with the provided inputs.
        
        Args:
            online_order (str): Whether the restaurant accepts online orders.
            book_table (str): Whether the restaurant allows table booking.
            votes (int): Number of votes the restaurant has received.
            rest_type (str): Type of restaurant.
            cost (float): Approximate cost for two people.
            type (str): Type of place (e.g., dine-out, delivery).
            city (str): City where the restaurant is located.
        """
        self.online_order = online_order
        self.book_table = book_table
        self.votes = votes
        self.rest_type = rest_type
        self.cost = cost
        self.type = type
        self.city = city
        logging.info("Initialized Prediction class with input data.")

    def get_dataframe(self):
        """
        Convert the input data into a DataFrame suitable for the model.
        
        Returns:
            pd.DataFrame: DataFrame containing the input data.
        """
        try:
            # Aggregate input data into a list
            data_points = [
                self.online_order,
                self.book_table,
                self.votes,
                self.rest_type,
                self.cost,
                self.type,
                self.city
            ]
            
            # Reshape the data to fit into a single row DataFrame
            dp = np.array(data_points).reshape(1, -1)
            logging.info(f"Reshaped data points for DataFrame: {dp}")
            
            # Define column names
            column_names = ["online_order", "book_table", "votes", "rest_type", "cost", "type", "city"]
            
            # Create DataFrame
            input_df = pd.DataFrame(dp, columns=column_names)
            logging.info("Created DataFrame for model input.")
            return input_df

        except Exception as e:
            logging.error(f"Error in get_dataframe: {e}")
            raise CustomException(e)

    def predict_rating(self, preprocessor_path, model_path):
        """
        Predict the rating using the pre-trained model and preprocessor.
        
        Args:
            preprocessor_path (str): Path to the preprocessor object file.
            model_path (str): Path to the trained model object file.
        
        Returns:
            float: Predicted rating.
        """
        try:
            # Load preprocessor and model
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            logging.info("Loaded preprocessor and model.")

            # Get input data as DataFrame
            new_df = self.get_dataframe()

            # Preprocess the input data
            new_df_array = preprocessor.transform(new_df)
            logging.info("Transformed input data using preprocessor.")

            # Predict the rating
            prediction = model.predict(new_df_array)
            logging.info(f"Model prediction: {prediction[0]}")
            return prediction[0]

        except Exception as e:
            logging.error(f"Error in predict_rating: {e}")
            raise CustomException(e)
