# Import necessary modules from Flask and custom pipeline
from flask import Flask, render_template, request
from source.pipeline.predict_pipeline import Prediction

# Create a Flask application instance
app = Flask(__name__)

# Define the home route
@app.route("/")
def home():
    """
    Render the homepage.

    Returns:
        Rendered HTML template for the homepage.
    """
    return render_template("index.html")

# Define the predict route that handles GET and POST requests
@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Handle prediction requests.

    If the request method is POST, retrieve form data, create a Prediction object,
    predict the rating, and render the result. If the request method is GET, render
    the input form.

    Returns:
        Rendered HTML template with prediction results or input form.
    """
    if request.method == "POST":
        # Retrieve form data
        online_order = request.form.get("online_order")
        book_table = request.form.get("book_table")
        votes = request.form.get("votes")
        rest_type = request.form.get("rest_type")
        cost = request.form.get("cost")
        type = request.form.get("type")
        city = request.form.get("city")
        
        # Create a Prediction object with the form data
        model_prediction_obj = Prediction(
            online_order=online_order,
            book_table=book_table,
            votes=votes,
            rest_type=rest_type,
            cost=cost,
            type=type,
            city=city
        )
        
        # Define paths to the model and preprocessor
        model_path = r"artifacts/model.pkl"
        preprocessor_path = r"artifacts/Preprocessor.pkl"
        
        # Predict the rating using the Prediction object
        output = model_prediction_obj.Predict_rating(model_path=model_path, preprocessor_path=preprocessor_path)
        
        # Function to convert rating to star format
        def rating_star(rate):
            """
            Convert a numerical rating to a star format string.
            
            Args:
                rate (float): The numerical rating.
            
            Returns:
                str: The rating in star format.
            """
            if rate == 2:
                return "⭐⭐"
            elif rate == 3:
                return "⭐⭐⭐"
            elif rate == 4:
                return "⭐⭐⭐⭐"
            elif rate == 5:
                return "⭐⭐⭐⭐⭐"
            else:
                return "⭐"
        
        # Get the rating in star format
        pred = rating_star(output)
        
        # Render the results page with the prediction
        return render_template("home.html", pred=pred)
    else:
        # If request method is GET, render the input form
        return render_template("index.html")

# Run the application
if __name__ == "__main__":
    print("Starting Flask application...")  # Debug print statement
    app.run(debug=True)
