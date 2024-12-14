from flask import Flask, render_template, request
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from datetime import datetime, timedelta
import io
import base64

app = Flask(__name__)

def fetch_stock_data(stock_symbol, period):
    """
    Fetch historical stock data for Indian markets using NSE (.NS) or BSE (.BO).
    """
    try:
        # Append .NS for NSE if no suffix is provided
        if not stock_symbol.endswith(('.NS', '.BO')):
            stock_symbol += ".NS"

        stock = yf.Ticker(stock_symbol)
        hist = stock.history(period="10y")  # Fetch 10 years of data for predictions

        if hist.empty:
            raise ValueError("No historical data available for this stock symbol.")

        # Filter data for the selected period
        if period == '1d':
            hist_period = hist.tail(1)
        elif period == '7d':
            hist_period = hist.tail(7)
        elif period == '1mo':
            hist_period = hist.tail(30)
        elif period == '1y':
            hist_period = hist.tail(252)
        else:
            hist_period = hist

        # Create a detailed plot
        img_base64 = plot_stock_data(hist_period, stock_symbol)

        # Get the latest stock details
        stock_details = {
            "Open": round(hist_period['Open'][-1], 2),
            "Close": round(hist_period['Close'][-1], 2),
            "High": round(hist_period['High'][-1], 2),
            "Low": round(hist_period['Low'][-1], 2),
            "Volume": int(hist_period['Volume'][-1])
        }

        # Generate predictions and recommendations
        predictions, recommendation = predict_stock_price(hist)

        return img_base64, stock_details, predictions, recommendation

    except Exception as e:
        print("Error fetching stock data:", e)
        return None, {}, None, None

def plot_stock_data(hist, stock_symbol):
    """
    Create a plot with stock Open, Close, High, Low, and Volume.
    """
    plt.figure(figsize=(12, 8))

    # Plot Price Data
    plt.subplot(2, 1, 1)
    plt.plot(hist['Close'], label="Close Price", color='blue', linewidth=2)
    plt.plot(hist['Open'], label="Open Price", color='green', linestyle='--')
    plt.fill_between(hist.index, hist['Low'], hist['High'], color='gray', alpha=0.2, label="High-Low Range")
    plt.title(f"{stock_symbol} Stock Prices")
    plt.ylabel("Price (₹)")  # Use ₹ for Rupees
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot Volume Data
    plt.subplot(2, 1, 2)
    plt.bar(hist.index, hist['Volume'], color='orange', alpha=0.7)
    plt.title(f"{stock_symbol} Volume")
    plt.ylabel("Volume")
    plt.xlabel("Date")

    plt.tight_layout()

    # Save plot to BytesIO
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    img_base64 = base64.b64encode(img.getvalue()).decode()
    return img_base64

def predict_stock_price(hist):
    """
    Predict future stock prices using Polynomial Regression and generate recommendation.
    """
    hist = hist.reset_index()
    hist['Date'] = pd.to_datetime(hist['Date'])
    hist['DateOrdinal'] = hist['Date'].map(lambda x: x.toordinal())

    X = hist['DateOrdinal'].values.reshape(-1, 1)
    y = hist['Close'].values

    # Polynomial Regression (Degree=3 for non-linear trends)
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    # Predict the next 5 days
    future_dates = [hist['Date'].max() + timedelta(days=i) for i in range(1, 6)]
    future_ordinal = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    future_poly = poly.transform(future_ordinal)
    predictions = model.predict(future_poly)

    prediction_dict = {future_dates[i].strftime('%Y-%m-%d'): round(predictions[i], 2) for i in range(5)}

    # Analyze trends
    recommendation = get_stock_recommendation(predictions)
    return prediction_dict, recommendation

def get_stock_recommendation(predictions):
    """
    Analyze stock predictions to recommend Buy, Sell, or Hold.
    """
    if predictions[0] < predictions[-1]:
        return "Buy"
    elif predictions[0] > predictions[-1]:
        return "Sell"
    else:
        return "Hold"

@app.route("/", methods=["GET", "POST"])
def index():
    img_base64 = None
    stock_details = {}
    predictions = None
    recommendation = None

    if request.method == "POST":
        stock_symbol = request.form.get("symbol").upper()
        period = request.form.get("period")

        img_base64, stock_details, predictions, recommendation = fetch_stock_data(stock_symbol, period)

    return render_template("index.html", 
                           img_base64=img_base64, 
                           stock_details=stock_details, 
                           predictions=predictions, 
                           recommendation=recommendation)

if __name__ == "__main__":
    app.run(debug=True)
