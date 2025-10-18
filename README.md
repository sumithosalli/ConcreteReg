# Concrete Strength Prediction Web Application

This project provides a web application for predicting the compressive strength of concrete. It leverages machine learning models trained on concrete mix composition and curing age data. The application includes a user-friendly interface for inputting parameters and viewing predictions, as well as a documented API endpoint.

## Features

*   **Concrete Strength Prediction:** Predicts concrete strength based on input parameters.
*   **Web Interface:** An interactive web page (`docs.html`, `index.html`, etc.) to input data and view results.
*   **API Endpoint:** A `POST /predict` endpoint for programmatic access to the prediction model.
*   **Model Performance Comparison:** Displays evaluation metrics for various regression models used during training.
*   **Trained Model Persistence:** The trained machine learning model is saved for efficient loading.

## Technologies Used

*   **Backend:** Python
*   **Machine Learning:** Scikit-learn, LightGBM, XGBoost, Gradient Boosting Regression, Linear Regression
*   **Data Handling:** Pandas
*   **Web Framework:** Flask (inferred from project structure)
*   **Dependencies:** joblib

## Project Structure

```
.
├── concrete_strength_model.pkl       # Trained machine learning model
├── main.py                          # Main application entry point (likely web server)
├── README.md                        # This file
├── .gitignore                       # Git ignore file
├── requirements.txt                 # Project dependencies
├── train_model.py                   # Script to train and save the ML model
├── data/                            # Directory for datasets
│   └── concrete_data.csv            # Training data
├── static/                          # Static assets (CSS, JS, images)
│   ├── about.css
│   ├── about.js
│   ├── blog.css
│   ├── contact.css
│   ├── docs.css                     # Styles for the documentation page
│   ├── index.css
│   ├── result.css
│   ├── services.css
│   └── images/                      # Image assets
└── templates/                       # HTML templates for the web interface
    ├── about.html
    ├── blog.html
    ├── contact.html
    ├── docs.html                    # Documentation page
    ├── index.html                   # Home page
    ├── result.html
    └── services.html
```

## Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sumithosalli/ConcreteReg.git
    cd ConcreteReg
    ```
2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Prepare the dataset:**
    Ensure the `data/concrete_data.csv` file is present. If not, place your dataset in this directory.
5.  **Train the machine learning model:**
    This script will train the model using the data and save it as `concrete_strength_model.pkl`.
    ```bash
    python train_model.py
    ```

## Usage

**1. Running the Web Application:**
Start the web server to access the application through your browser.
```bash
python main.py
```
Then, open your web browser and navigate to `http://127.0.0.1:5000/` (or the address provided by the server).

**2. Using the API Endpoint:**
You can send a POST request to the `/predict` endpoint with the required parameters to get a concrete strength prediction.

*   **Endpoint:** `POST /predict`
*   **Parameters:**
    *   `cement` (float): Amount of cement (kg/m³)
    *   `blast_furnace_slag` (float): Blast furnace slag (kg/m³)
    *   `fly_ash` (float): Fly ash (kg/m³)
    *   `water` (float): Water (kg/m³)
    *   `superplasticizer` (float): Superplasticizer (kg/m³)
    *   `coarse_aggregate` (float): Coarse aggregate (kg/m³)
    *   `age` (int): Curing age (days)

**Example Request (using `curl`):**
```bash
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{
    "cement": 300,
    "blast_furnace_slag": 100,
    "fly_ash": 50,
    "water": 150,
    "superplasticizer": 10,
    "coarse_aggregate": 800,
    "age": 7
}'
```

## Model Performance Comparison

The following table summarizes the performance metrics of different models evaluated during training:

| Model Name                   | Mean Absolute Error (MAE) | Mean Squared Error (MSE) | Root Mean Squared Error (RMSE) | R² Score |
| :--------------------------- | :------------------------ | :----------------------- | :----------------------------- | :------- |
| Linear Regression            | 8.36                      | 109.31                   | 10.46                          | 0.3386   |
| Random Forest                | 3.71                      | 25.67                    | 5.07                           | 0.8967   |
| Gradient Boosting Regression | 3.55                      | 20.98                    | 4.58                           | 0.9275   |
| XGBoost                      | 2.20                      | 8.98                     | 3.00                           | 0.9689   |
| LGBMRegressor                | 2.21                      | 8.66                     | 2.94                           | 0.9701   |

The `XGBoost` and `LGBMRegressor` models demonstrate superior performance with the lowest error metrics and highest R² scores.

## Contact

Created by [@sumithosalli](https://github.com/sumithosalli)
