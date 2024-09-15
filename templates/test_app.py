import unittest
from flask_testing import TestCase
import pandas as pd
import numpy as np
import os
from app import app, generate_data, load_data, preprocess_data, train_test_split_ts, create_sequences, arima_forecast, train_ml_models, train_lstm, plot_to_base64

class AppTestCase(TestCase):

    def create_app(self):
        app.config['TESTING'] = True
        return app

    def test_generate_data(self):
        file_path = generate_data()
        self.assertTrue(file_path.endswith('.csv'))
        self.assertTrue(os.path.exists(file_path))

    def test_load_data(self):
        file_path = generate_data()
        data = load_data(file_path)
        self.assertIsInstance(data, pd.DataFrame)

    def test_preprocess_data(self):
        file_path = generate_data()
        data = load_data(file_path)
        data = preprocess_data(data)
        self.assertIn('Customer_Profile', data.columns)
        self.assertIn('Product_Preference', data.columns)
        self.assertIn('Channel_Performance', data.columns)

    def test_train_test_split_ts(self):
        file_path = generate_data()
        data = load_data(file_path)
        train, test = train_test_split_ts(data)
        self.assertEqual(len(train) + len(test), len(data))

    def test_create_sequences(self):
        file_path = generate_data()
        data = load_data(file_path)
        scaled_data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data[['Value']])
        X, y = create_sequences(scaled_data, 12)
        self.assertEqual(X.shape[1], 12)

    def test_arima_forecast(self):
        file_path = generate_data()
        data = load_data(file_path)
        train, test = train_test_split_ts(data)
        arima_forecast_values, arima_mse = arima_forecast(train['Value'].values, test['Value'].values)
        self.assertEqual(len(arima_forecast_values), len(test))

    def test_train_ml_models(self):
        file_path = generate_data()
        data = load_data(file_path)
        data = preprocess_data(data)
        train, test = train_test_split_ts(data)
        scaled_data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data[['Value']])
        X, y = create_sequences(scaled_data, 12)
        X_2D = X.reshape((X.shape[0], X.shape[1]))
        X_train, X_test, y_train, y_test = train_test_split(X_2D, y, test_size=0.2, shuffle=False)
        rf_mse, gb_mse, stack_mse = train_ml_models(X_train, y_train, X_test, y_test)
        self.assertIsInstance(rf_mse, float)
        self.assertIsInstance(gb_mse, float)
        self.assertIsInstance(stack_mse, float)

    def test_train_lstm(self):
        file_path = generate_data()
        data = load_data(file_path)
        data = preprocess_data(data)
        train, test = train_test_split_ts(data)
        scaled_data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data[['Value']])
        X, y = create_sequences(scaled_data, 12)
        X_2D = X.reshape((X.shape[0], X.shape[1]))
        X_train, X_test, y_train, y_test = train_test_split(X_2D, y, test_size=0.2, shuffle=False)
        lstm_pred, lstm_mse = train_lstm(X_train, y_train, X_test, y_test, MinMaxScaler(feature_range=(0, 1)))
        self.assertEqual(len(lstm_pred), len(y_test))

    def test_plot_to_base64(self):
        file_path = generate_data()
        data = load_data(file_path)
        scaled_data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data[['Value']])
        X, y = create_sequences(scaled_data, 12)
        arima_forecast_values, _ = arima_forecast(data['Value'].values[:len(data) // 2], data['Value'].values[len(data) // 2:])
        plot_img = plot_to_base64(data, arima_forecast_values, 'ARIMA Forecast')
        self.assertTrue(plot_img.startswith('iVBORw0KGgo='))

    def test_generate_data_route(self):
        response = self.client.get('/generate_data')
        self.assertEqual(response.status_code, 200)
        data = response.json
        self.assertIn('file_path', data)
        self.assertTrue(os.path.exists(data['file_path']))

    def test_upload_file(self):
        with open('data/enhanced_time_series_data.csv', 'rb') as file:
            response = self.client.post('/upload', data={'file': (file, 'test_file.csv')})
        self.assertEqual(response.status_code, 200)
        data = response.json
        self.assertIn('file_path', data)
        self.assertTrue(os.path.exists(data['file_path']))

    def test_train_models_route(self):
        file_path = generate_data()
        response = self.client.post('/train_models', data={'file_path': file_path})
        self.assertEqual(response.status_code, 200)
        data = response.json
        self.assertIn('arima_mse', data)
        self.assertIn('lstm_mse', data)
        self.assertIn('arima_plot', data)
        self.assertIn('lstm_plot', data)
        self.assertTrue(data['arima_plot'].startswith('iVBORw0KGgo='))
        self.assertTrue(data['lstm_plot'].startswith('iVBORw0KGgo='))

if __name__ == '__main__':
    unittest.main()
