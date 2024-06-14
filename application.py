import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from source.pipeline.predict_pipeline import CustomData, PredictPipeline

# Set page config for a custom icon and title
st.set_page_config(page_title="Zomato Rating Prediction", page_icon="🍽️")

# Apply custom CSS
# Apply custom CSS for better styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f4f4f9;
        padding: 2rem;
    }
    .header {
        background-color: #d9534f;
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    .header h1 {
        margin: 0;
        font-size: 2.5rem;
    }
    .header h3 {
        margin: 0;
        font-size: 1.5rem;
    }
    .stSelectbox > div {
        cursor: pointer;
    }
    .stButton>button {
        background-color: #d9534f;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        border: 1px solid #d43f3a;
        padding: 0.5em 1em;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #c9302c;
        border-color: #ac2925;
    }
    .stTitle, .stMarkdown {
        text-align: center;
    }
    .stSuccess {
        color: #28a745;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
    }
    .author {
        font-size: 14px;
        color: grey;
        text-align: left;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom header
st.markdown(
    """
    <div class="header">
        <h1>Zomato Rating Prediction App</h1>
        <h3>Predict the Rating of a Restaurant</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# Title of the web app
#st.title("Zomato Rating Prediction")

# Description of the web app
#st.markdown("### Fill the fields below to predict the rating of a restaurant.")

# Author label in the sidebar
#st.sidebar.markdown("<div class='author'>Author: Lavish Gangwani</div>", unsafe_allow_html=True)

# Collect user input
online_order = st.selectbox("Online Order", ["Select an option", "Yes", "No"])
book_table = st.selectbox("Book Table", ["Select an option", "Yes", "No"])
votes = st.slider("Votes", min_value=0, max_value=9000, step=1)
rest_type = st.selectbox("Rest Type", [
    "Select an option", "Quick Bites", "Casual Dining", "Cafe", "Other Rest Type", "Delivery",
    "Dessert Parlor", "Takeaway, Delivery", "Casual Dining, Bar", "Bakery",
    "Beverage Shop", "Bar", "Food Court", "Sweet Shop", "Bar, Casual Dining",
    "Lounge", "Pub", "Fine Dining", "Casual Dining, Cafe", "Beverage Shop, Quick Bites",
    "Bakery, Quick Bites", "Mess", "Pub, Casual Dining"
])
cost = st.number_input("Cost", min_value=0, step=1)
type = st.selectbox("Type", ["Select an option", "Delivery", "Dine-out", "Desserts", "Cafes", "Drinks & nightlife", "Buffet", "Pubs and bars"])
city = st.selectbox("City", [
    "Select an option", "Banashankari", "Bannerghatta Road", "Basavanagudi", "Bellandur", "Brigade Road",
    "Brookefield", "BTM", "Church Street", "Electronic City", "Frazer Town",
    "HSR", "Indiranagar", "Jayanagar", "JP Nagar", "Kalyan Nagar", "Kammanahalli",
    "Koramangala 4th Block", "Koramangala 5th Block", "Koramangala 6th Block",
    "Koramangala 7th Block", "Lavelle Road", "Malleshwaram", "Marathahalli",
    "MG Road", "New BEL Road", "Old Airport Road", "Rajajinagar", "Residency Road",
    "Sarjapur Road", "Whitefield"
])

# Check if all fields are filled before predicting
if st.button("Predict Rating"):
    # Check if all fields are filled
    if online_order != "Select an option" and book_table != "Select an option" and rest_type != "Select an option" and type != "Select an option" and city != "Select an option" and cost != 0 and votes != 0:
        # Create a CustomData instance with the input values
        custom_data = CustomData(
            online_order=online_order,
            book_table=book_table,
            votes=votes,
            rest_type=rest_type,
            cost=cost,
            type=type,
            city=city
        )

        # Get the DataFrame representation of the input data
        data = custom_data.get_data_as_data_frame()

        # Create a PredictPipeline instance and make a prediction
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(data)

        # Display the prediction
        st.success(f"The predicted rating of the restaurant is: {prediction[0]}")
    else:
        st.warning("Please fill all fields.")

# Footer
st.markdown("##### Author : Lavish Gangwani")
