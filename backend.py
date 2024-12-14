import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from datetime import datetime, timedelta

def fetch_stock_data(stock_symbol, period):
    """
    Fetch historical stock data for the given period and display stock information.
    """
    try:
        # Fetch stock data for the given symbol
        stock = yf.Ticker(stock_symbol)

        # Fetch the last 10 years of data for prediction
        hist = stock.history(period="10y")  # Last 10 years data for prediction

        # Ensure 'Close' column exists in the DataFrame
        if 'Close' not in hist.columns:
            raise ValueError("'Close' column not found in the historical data. Verify stock symbol or data fetch process.")

        # Display stock information
        print(f"\nStock Symbol: {stock_symbol}")
        print(f"Company Name: {stock.info.get('longName', 'N/A')}")
        print(f"Current Price: {stock.info.get('currentPrice', 'N/A')} {stock.info.get('currency', 'N/A')}")
        print(f"Market Cap: {stock.info.get('marketCap', 'N/A')}")
        print(f"52 Week High: {stock.info.get('fiftyTwoWeekHigh', 'N/A')}")
        print(f"52 Week Low: {stock.info.get('fiftyTwoWeekLow', 'N/A')}")
        print(f"Total Records Fetched: {len(hist)}")

        # Handle the period for the graph
        if period == '1d':
            hist_for_graph = hist.tail(1)  # Last 1 day
            intraday_prediction(stock_symbol)
        elif period == '7d':
            hist_for_graph = hist.tail(7)  # Last 1 week (7 days)
        elif period == '1mo':
            hist_for_graph = hist.tail(30)  # Last 1 month (30 days)
        elif period == '1y':
            hist_for_graph = hist.tail(252)  # Last 1 year (~252 trading days)

        # Display the fetched data (for the user)
        print("\nHistorical Data for Selected Period (Graph):")
        print(hist_for_graph)

        # Plot the data for the selected period (1d, 7d, 1mo, 1y)
        plt.figure(figsize=(12, 6))
        plt.plot(hist_for_graph['Close'], label=f"{stock_symbol} Closing Price", color='blue', linewidth=2)
        plt.title(f"Stock Price Trend for {stock_symbol} ({period} Data)", fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Closing Price", fontsize=12)
        plt.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Predict stock prices using polynomial regression based on the last 10 years of data
        predict_stock_price(hist, stock, hist_for_graph, period)

    except Exception as e:
        print("Error fetching stock data:", e)

def intraday_prediction(stock_symbol):
    """
    Fetch intraday data for 1 day and predict the next 4-5 hours.
    """
    stock = yf.Ticker(stock_symbol)

    # Fetch the last day of intraday data with 1-hour intervals
    intraday_hist = stock.history(period="1d", interval="1h")  # 1-day, hourly data
    print("\nIntraday Data (1-Day, 1 Hour Interval):")
    print(intraday_hist)

    # Prepare data for intraday prediction
    intraday_hist = intraday_hist.reset_index()
    intraday_hist['DateOrdinal'] = intraday_hist['Datetime'].map(lambda x: x.toordinal())  # Convert dates to ordinal

    # Features (X) and Target (y)
    X = intraday_hist['DateOrdinal'].values.reshape(-1, 1)  # Date as input
    y = intraday_hist['Close'].values  # Closing price as target

    # Polynomial Regression Setup (Degree = 3 for non-linear trends)
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    # Predict the next 4-5 hours (using next few hours)
    future_hours = [intraday_hist['Datetime'].max() + timedelta(hours=i) for i in range(1, 6)]  # Next 5 hours
    future_ordinal = np.array([d.toordinal() for d in future_hours]).reshape(-1, 1)
    future_poly = poly.transform(future_ordinal)
    predictions = model.predict(future_poly)

    # Print future predictions for the next 4-5 hours
    print("\nIntraday Predictions for the Next 4-5 Hours:")
    for date, pred in zip(future_hours, predictions):
        print(f"{date.strftime('%Y-%m-%d %H:%M:%S')}: {pred:.2f}")

    # Suggest Buy/Sell for Intraday
    current_price = y[-1]
    avg_future_price = np.mean(predictions)
    print("\nIntraday Stock Analysis:")
    if avg_future_price > current_price * 1.01:  # If price increases by 1% over the next few hours
        print("Suggestion: BUY (Expected Increase in Price)")
    elif avg_future_price < current_price * 0.99:  # If price decreases by 1% over the next few hours
        print("Suggestion: SELL (Expected Decrease in Price)")
    else:
        print("Suggestion: HOLD (Minimal Price Change Expected)")

    # Plot historical intraday data and predictions for next 5 hours
    plt.figure(figsize=(12, 6))
    plt.plot(intraday_hist['Datetime'], intraday_hist['Close'], label=f"Intraday Closing Price", color='blue')

    # Plot predicted data for the next 5 hours
    plt.plot(future_hours, predictions, label="Predicted Closing Price (Next 5 Hours)", color='red', linestyle='--')

    # Add text annotations for predicted values
    for date, pred in zip(future_hours, predictions):
        plt.text(date, pred, f"{pred:.2f}", color='red', fontsize=9, ha='center')

    # Titles and labels
    plt.title(f"Intraday Price Trend and 5-Hour Prediction for {stock_symbol}", fontsize=16)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Closing Price", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def predict_stock_price(hist, stock, hist_for_graph, period):
    """
    Predict future stock prices using Polynomial Regression and suggest Buy/Sell.
    """
    # Reset index and prepare data for polynomial regression (based on the last 10 years)
    hist = hist.reset_index()
    hist['Date'] = pd.to_datetime(hist['Date'])
    hist['DateOrdinal'] = hist['Date'].map(lambda x: x.toordinal())  # Convert dates to ordinal

    # Features (X) and Target (y)
    X = hist['DateOrdinal'].values.reshape(-1, 1)  # Date as input
    y = hist['Close'].values  # Closing price as target

    # Polynomial Regression Setup (Degree = 3 for non-linear trends)
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    # Predict the next 5 days
    future_dates = [hist['Date'].max() + timedelta(days=i) for i in range(1, 6)]  # Next 5 days
    future_ordinal = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    future_poly = poly.transform(future_ordinal)
    predictions = model.predict(future_poly)

    # Print future predictions
    print("\nPredictions for the Next 5 Days:")
    for date, pred in zip(future_dates, predictions):
        print(f"{date.strftime('%Y-%m-%d')}: {pred:.2f}")

    # Analyze trend and suggest Buy/Sell
    current_price = y[-1]
    avg_future_price = np.mean(predictions)
    print("\nStock Analysis:")
    if avg_future_price > current_price * 1.02:  # If future price increases by 2%
        print("Suggestion: BUY (Expected Increase in Price)")
    elif avg_future_price < current_price * 0.98:  # If future price decreases by 2%
        print("Suggestion: SELL (Expected Decrease in Price)")
    else:
        print("Suggestion: HOLD (Minimal Price Change Expected)")

    # Plot historical data and predictions (10 years data for prediction and 5-day prediction)
    plt.figure(figsize=(12, 6))

    # Plot historical data for the selected period
    plt.plot(hist_for_graph['Close'], label=f"Historical {period} Data", color='blue')

    # Plot predicted data for the next 5 days
    plt.plot(future_dates, predictions, label="Predicted Closing Price (Next 5 Days)", color='red', linestyle='--')

    # Add text annotations for predicted values
    for date, pred in zip(future_dates, predictions):
        plt.text(date, pred, f"{pred:.2f}", color='red', fontsize=9, ha='center')

    # Titles and labels
    plt.title(f"Stock Price Trend and 5-Day Prediction for {stock.info.get('longName', 'N/A')}", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Closing Price", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to select stock symbol and period.
    """
    print("\nSelect Time Period for Stock Data:")
    print("1. 1 Day")
    print("2. 1 Week")
    print("3. 1 Month")
    print("4. 1 Year")
    choice = input("Enter your choice (1/2/3/4): ").strip()

    period_map = {
        "1": "1d",
        "2": "7d",
        "3": "1mo",
        "4": "1y"
    }

    period = period_map.get(choice, "1mo")
    symbol = input("Enter Stock Symbol (e.g., TCS.NS, INFY.NS, RELIANCE.NS): ").upper()
    fetch_stock_data(symbol, period)

if __name__ == "__main__":
    main()
