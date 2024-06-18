from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from source.pipeline.predict_pipeline import CustomData, PredictPipeline
from source.logger import logging

app = FastAPI()


# Pydantic model for input data validation
class RatingInput(BaseModel):
    online_order: str
    book_table: str
    votes: int
    rest_type: str
    cost: float
    type: str
    city: str

# Define valid options
valid_online_order = ["Yes", "No"]
valid_book_table = ["Yes", "No"]
valid_rest_type = [
    "Quick Bites", "Casual Dining", "Cafe", "Other Rest Type", "Delivery",
    "Dessert Parlor", "Takeaway, Delivery", "Casual Dining, Bar", "Bakery",
    "Beverage Shop", "Bar", "Food Court", "Sweet Shop", "Bar, Casual Dining",
    "Lounge", "Pub", "Fine Dining", "Casual Dining, Cafe", "Beverage Shop, Quick Bites",
    "Bakery, Quick Bites", "Mess", "Pub, Casual Dining"
]
valid_type = ["Delivery", "Dine-out", "Desserts", "Cafes", "Drinks & nightlife", "Buffet", "Pubs and bars"]
valid_city = [
    "Banashankari", "Bannerghatta Road", "Basavanagudi", "Bellandur", "Brigade Road",
    "Brookefield", "BTM", "Church Street", "Electronic City", "Frazer Town",
    "HSR", "Indiranagar", "Jayanagar", "JP Nagar", "Kalyan Nagar", "Kammanahalli",
    "Koramangala 4th Block", "Koramangala 5th Block", "Koramangala 6th Block",
    "Koramangala 7th Block", "Lavelle Road", "Malleshwaram", "Marathahalli",
    "MG Road", "New BEL Road", "Old Airport Road", "Rajajinagar", "Residency Road",
    "Sarjapur Road", "Whitefield"
]

@app.post("/predict")
async def predict_rating(input_data: RatingInput):
    try:
        # Validate input values
        if input_data.online_order not in valid_online_order:
            raise HTTPException(status_code=400, detail="Invalid value for online_order")
        if input_data.book_table not in valid_book_table:
            raise HTTPException(status_code=400, detail="Invalid value for book_table")
        if input_data.rest_type not in valid_rest_type:
            raise HTTPException(status_code=400, detail="Invalid value for rest_type")
        if input_data.type not in valid_type:
            raise HTTPException(status_code=400, detail="Invalid value for type")
        if input_data.city not in valid_city:
            raise HTTPException(status_code=400, detail="Invalid value for city")
        if input_data.cost <= 0:
            raise HTTPException(status_code=400, detail="Cost must be greater than 0")

        # Create a CustomData instance with the input values
        custom_data = CustomData(
            online_order=input_data.online_order,
            book_table=input_data.book_table,
            votes=input_data.votes,
            rest_type=input_data.rest_type,
            cost=input_data.cost,
            type=input_data.type,
            city=input_data.city
        )

        # Log input data
        logging.info(f"Input data: {custom_data}")

        # Get the DataFrame representation of the input data
        data = custom_data.get_data_as_data_frame()

        # Log DataFrame
        logging.info(f"DataFrame: {data}")

        # Create a PredictPipeline instance and make a prediction
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(data)

        # Log prediction
        logging.info(f"Prediction: {prediction}")

        return {"predicted_rating": prediction[0]}
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Zomato Rating Prediction API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
