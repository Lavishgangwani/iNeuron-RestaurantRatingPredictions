import sys
import pandas as pd
from source.exception import CustomException
from source.utils import load_object
import os
from source.logger import logging


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,online_order,book_table,votes,rest_type,cost,type,city):
        self.online_order=online_order
        self.book_table=book_table
        self.votes=votes
        self.rest_type=rest_type
        self.cost=cost
        self.type=type
        self.city=city
        logging.info("get all the data")

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "online_order": [self.online_order],
                "book_table": [self.book_table],
                "votes": [self.votes],
                "rest_type": [self.rest_type],
                "cost": [self.cost],
                "type": [self.type],
                "city": [self.city],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
        