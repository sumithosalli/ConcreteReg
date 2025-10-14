# Concrete Strength Prediction

A simple machine learning project to predict concrete strength based on its ingredients.

## Quick Start

1. Install required packages:

```bash
pip install pandas scikit-learn lightgbm joblib
```

1. Put your `archive.zip` (containing Concrete_Data.csv) in the project folder

1. Run the model:

```bash
python ConcreteReg/train_model.py
```

## What it Does

- Takes concrete mixture information (cement, water, aggregates, etc.)
- Predicts the concrete's strength
- Uses LightGBM for accurate predictions
- Shows how well the predictions worked (using MAE, MSE, RMSE, R²)

## Input Features

- Cement
- Blast Furnace Slag
- Fly Ash
- Water
- Superplasticizer
- Coarse Aggregate
- Age

## Project Structure

```txt
ConcreteReg/
├── data/           # Your dataset goes here (auto-created)
├── train_model.py  # Main script
└── README.md      # This file
```

## Contact

Created by [@sumithosalli](https://github.com/sumithosalli)