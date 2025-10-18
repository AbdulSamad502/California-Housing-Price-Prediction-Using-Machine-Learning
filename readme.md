## California Price Prediction

#About

This project predicts median house prices using machine learning on the California Housing dataset. It uses pipelines to preprocess numerical and categorical features, splits data using stratified sampling, and trains a RandomForestRegressor. The trained model and preprocessing pipeline are saved for inference, allowing predictions on new data.

## Features
Automatic data preprocessing with Pipeline and ColumnTransformer.
Handles missing values and scales numerical features.
One-hot encoding for categorical features (ocean_proximity).
Stratified train-test split based on income categories.
Model training using RandomForestRegressor.
Model persistence with joblib for future inference.

## Installation
Clone the repository:
git clone <your-repo-url>
cd <repo-folder>


## Install required packages:
pip install -r requirements.txt

## Usage
Run the script:
python main.py
Predictions will be generated automatically and saved to output.csv.

## Files
main.py – Main script for training and inference.
model.pkl – Trained machine learning model.
pipeline.pkl – Data preprocessing pipeline.

## Author
Abdul Samad
Email:-samad784600@gmail.com
GitHub:-AbdulSamad502



