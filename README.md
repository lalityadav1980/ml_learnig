# House Price Prediction

This project aims to predict house prices based on various features such as the size of the house, number of bedrooms, and location. We utilize linear regression as our modeling technique to achieve this.

## Datasets

1. **Boston Housing Dataset**: Located in `data/boston_housing.csv`, this dataset includes features like the size of the house, number of bedrooms, and location, along with the target variable (house price).

2. **Kaggle House Price Dataset**: Found in `data/kaggle_house_price.csv`, this dataset provides additional features and target values for house prices.

## Project Structure

```
house-price-prediction
├── data
│   ├── boston_housing.csv
│   └── kaggle_house_price.csv
├── notebooks
│   └── linear_regression.ipynb
├── src
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── model_evaluation.py
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd house-price-prediction
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Notebook

To run the linear regression model, open the Jupyter notebook located in `notebooks/linear_regression.ipynb`. This notebook includes sections for:

- Data loading
- Data preprocessing
- Model training
- Model evaluation

## Evaluation Metrics

The model's performance will be evaluated using the following metrics:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R-squared (R²)

## License

This project is licensed under the MIT License.