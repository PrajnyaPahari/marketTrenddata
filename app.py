from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os

app = Flask(__name__)

# Utility functions
def generate_data(num_samples=10000):
    np.random.seed(0)
    start_date = '2010-01-01'
    end_date = pd.to_datetime(start_date) + pd.DateOffset(days=num_samples - 1)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[:num_samples]
    
    seasonal_pattern = 10 + 5 * np.sin(np.linspace(0, 2 * np.pi * len(dates) / 12, len(dates)))
    trend = np.linspace(0, 100, len(dates))
    noise = np.random.normal(scale=10, size=len(dates))
    values = seasonal_pattern + trend + noise

    customer_profiles = np.random.choice(['Low', 'Medium', 'High'], len(dates))
    product_preferences = np.random.choice(['A', 'B', 'C'], len(dates))
    campaign_success = np.random.choice([0, 1], len(dates))
    channel_performance = np.random.choice(['Email', 'Social Media', 'Ads'], len(dates))
    
    data = pd.DataFrame({
        'Date': dates,
        'Value': values,
        'Customer_Profile': customer_profiles,
        'Product_Preference': product_preferences,
        'Campaign_Success': campaign_success,
        'Channel_Performance': channel_performance
    })
    
    file_path = 'data/enhanced_time_series_data.csv'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    data.to_csv(file_path, index=False)
    return file_path

def load_data(file_path):
    try:
        data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        data_resampled = data.select_dtypes(include=[np.number]).resample('ME').mean().ffill()
        non_numeric = data.select_dtypes(exclude=[np.number]).resample('ME').first()
        data_combined = data_resampled.join(non_numeric)
        return data_combined
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")

def preprocess_data(data):
    label_enc = LabelEncoder()
    for column in ['Customer_Profile', 'Product_Preference', 'Channel_Performance']:
        data[column] = label_enc.fit_transform(data[column])
    return data

def train_test_split_ts(data, train_size=0.8):
    train_size = int(len(data) * train_size)
    train, test = data[:train_size], data[train_size:]
    return train, test

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def arima_forecast(train, test):
    try:
        arima_model = ARIMA(train, order=(5,1,0))
        arima_fit = arima_model.fit()
        arima_forecast = arima_fit.forecast(steps=len(test))
        arima_forecast_flat = arima_forecast.ravel()
        test_flat = test.ravel()
        arima_mse = mean_squared_error(test_flat, arima_forecast_flat)
        return arima_forecast_flat, arima_mse
    except Exception as e:
        raise RuntimeError(f"Error in ARIMA forecasting: {e}")

def train_ml_models(X_train, y_train, X_test, y_test):
    try:
        rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_mse = mean_squared_error(y_test, rf_pred)

        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=0)
        gb_model.fit(X_train, y_train)
        gb_pred = gb_model.predict(X_test)
        gb_mse = mean_squared_error(y_test, gb_pred)

        base_models = [
            ('rf', RandomForestRegressor(n_estimators=100, random_state=0)),
            ('gb', GradientBoostingRegressor(n_estimators=100, random_state=0))
        ]
        meta_model = GradientBoostingRegressor(n_estimators=50, random_state=0)
        stack_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)
        stack_model.fit(X_train, y_train)
        stack_pred = stack_model.predict(X_test)
        stack_mse = mean_squared_error(y_test, stack_pred)

        return rf_mse, gb_mse, stack_mse
    except Exception as e:
        raise RuntimeError(f"Error training ML models: {e}")


def train_lstm(X_train, y_train, X_test, y_test, scaler):
    try:
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        lstm_model = Sequential()
        lstm_model.add(tf.keras.Input(shape=(X_train.shape[1], 1)))  # Update this line
        lstm_model.add(LSTM(50, activation='relu'))
        lstm_model.add(Dense(1))
        lstm_model.compile(optimizer='adam', loss='mse')

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stopping],
                       verbose=1)

        lstm_pred = lstm_model.predict(X_test)
        lstm_pred = scaler.inverse_transform(lstm_pred)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        lstm_pred_flat = lstm_pred.ravel()
        y_test_flat = y_test.ravel()

        lstm_mse = mean_squared_error(y_test_flat, lstm_pred_flat)

        return lstm_pred, lstm_mse
    except Exception as e:
        raise RuntimeError(f"Error training LSTM model: {e}")


def perform_clustering(data, n_clusters=3):
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        clusters = kmeans.fit_predict(data)
        return clusters, kmeans
    except Exception as e:
        raise RuntimeError(f"Error performing clustering: {e}")

def perform_pca(data, n_components=2):
    try:
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(data)
        return transformed_data, pca
    except Exception as e:
        raise RuntimeError(f"Error performing PCA: {e}")

def plot_to_base64(data, forecast_values, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    data['Value'].plot(ax=ax, label='Actual')
    forecast_series = pd.Series(forecast_values.flatten(), index=data.index[-len(forecast_values):])
    forecast_series.plot(ax=ax, label=title)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close(fig)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_data')
def generate_data_route():
    try:
        file_path = generate_data()
        return jsonify({'file_path': file_path})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    file_path = 'data/uploaded_data.csv'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file.save(file_path)
    return jsonify({'file_path': file_path})

@app.route('/train_models', methods=['POST'])
def train_models():
    file_path = request.form.get('file_path')
    if not file_path:
        return jsonify({'error': 'Please upload a CSV file first.'}), 400
    
    try:
        data = load_data(file_path)
        data = preprocess_data(data)
        train, test = train_test_split_ts(data)

        global seq_length, scaler
        seq_length = 12
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[['Value']])
        X, y = create_sequences(scaled_data, seq_length)

        X_2D = X.reshape((X.shape[0], X.shape[1]))
        X_train, X_test, y_train, y_test = train_test_split(X_2D, y, test_size=0.2, shuffle=False)

        arima_forecast_values, arima_mse = arima_forecast(train['Value'].values, test['Value'].values)
        rf_mse, gb_mse, stack_mse = train_ml_models(X_train, y_train, X_test, y_test)
        lstm_pred, lstm_mse = train_lstm(X_train, y_train, X_test, y_test, scaler)
        
        # Plotting
        arima_plot = plot_to_base64(data, arima_forecast_values, 'ARIMA Forecast')
        lstm_plot = plot_to_base64(data, lstm_pred, 'LSTM Forecast')

        return jsonify({
            'arima_mse': arima_mse,
            'rf_mse': rf_mse,
            'gb_mse': gb_mse,
            'stack_mse': stack_mse,
            'lstm_mse': lstm_mse,
            'arima_plot': arima_plot,
            'lstm_plot': lstm_plot
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)