import unittest
from app import app, generate_data, load_data, preprocess_data, train_test_split_ts, create_sequences, arima_forecast, \
    train_ml_models, train_lstm
from sklearn.preprocessing import MinMaxScaler  # Add this import
import pandas as pd
import numpy as np


class TestApp(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_generate_data(self):
        file_path = generate_data()
        self.assertTrue(file_path.endswith('.csv'))

    def test_load_data(self):
        file_path = generate_data()
        data = load_data(file_path)
        self.assertFalse(data.empty)

    def test_preprocess_data(self):
        file_path = generate_data()
        data = load_data(file_path)
        processed_data = preprocess_data(data)
        self.assertIn('Customer_Profile', processed_data.columns)
        self.assertIn('Product_Preference', processed_data.columns)
        self.assertIn('Channel_Performance', processed_data.columns)

    def test_create_sequences(self):
        file_path = generate_data()
        data = load_data(file_path)
        data = preprocess_data(data)
        scaler = MinMaxScaler(feature_range=(0, 1))  # Now it's defined
        scaled_data = scaler.fit_transform(data[['Value']])
        seq_length = 12
        X, y = create_sequences(scaled_data, seq_length)
        self.assertEqual(X.shape[1], seq_length)
        self.assertEqual(X.shape[0], y.shape[0])

    def test_arima_forecast(self):
        file_path = generate_data()
        data = load_data(file_path)
        data = preprocess_data(data)
        train, test = train_test_split_ts(data)
        forecast_values, mse = arima_forecast(train['Value'].values, test['Value'].values)
        self.assertEqual(len(forecast_values), len(test))


if __name__ == '__main__':
    unittest.main()