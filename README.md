
---

# Zomato Bangalore Restaurants Rating Prediction

![Project Screenshot](https://user-images.githubusercontent.com/92681972/233918490-f22e93c9-49fa-40a8-8e76-ad996c29be70.png)

This project predicts restaurant ratings in Bangalore using machine learning techniques.

## Overview

This repository contains the code for predicting restaurant ratings in Bangalore using various machine learning models. The project involves data ingestion, preprocessing, model training, and deployment of a web application for user interaction.

## Libraries Used

The project relies on several Python libraries, all of which are listed in `requirements.txt`. To install the dependencies, run:

```bash
pip install -r requirements.txt
```

- `pandas`: Data manipulation and analysis.
- `numpy`: Numerical operations on arrays and matrices.
- `seaborn`, `matplotlib`: Data visualization libraries.
- `scikit-learn`: Machine learning toolkit for data mining and analysis.
- `xgboost`: Gradient boosting library for optimized distributed gradient boosting.
- `flask`: Micro web framework for building web applications in Python.
- `dill`: Serialization library for Python objects.
- `streamlit`: Open-source app framework for machine learning and data science projects.

## Project Structure

### Files and Directories

- `setup.py`: Project setup details.
- `requirements.txt`: Dependency list for easy installation.
- `source/logger.py`: Logging configuration for the project.
- `source/exception.py`: Custom exception handling for error tracking.
- `.gitignore`: Specifies files and directories ignored by Git.
- `README.md`: General project information and setup instructions.
- `data/`: Directory for dataset storage.
- `source/`: Source code directory.
  - `__init__.py`: Enables module usage.
  - `source/components/data_ingestion.py`: Manages data ingestion from various sources.
  - `source/components/data_transformation.py`: Handles data preprocessing and transformation.
  - `source/components/model_trainer.py`: Trains machine learning models and performs hyperparameter tuning.
  - `source/pipeline/prediction_pipeline.py`: Creates a web application using `app.py` and `utils.py`.
  - `source/utils.py`: Stores common functions used throughout the project.
  - `application.py`: Streamlit application for user interaction and prediction.

## 🚀 Run Locally

To run this project locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/Lavishgangwani/iNeuron-RestaurantRatingPredictions.git
   ```

2. Navigate into the project directory:

   ```bash
   cd iNeuron-RestaurantRatingPredictions
   ```

3. Create and activate a virtual environment:

   ```bash
   python -m venv myenv
   myenv\Scripts\activate (Windows)
   source myenv/bin/activate (Mac/Linux)
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

6. Open a web browser and go to the local Streamlit URL provided after running the above command to use the application locally.

## Deployment

The project is deployed using Streamlit and can be accessed publicly via the following link: [Zomato Bangalore Restaurants Prediction App](https://ineuron-restaurantratingpredictions-lavishgangwani22.streamlit.app/).

## 🎯 Project Created by

[Lavish Gangwani](https://linkedin.com/in/lavish-gangwani)  
Email: lavishgangwani22@gmail.com

---