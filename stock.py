import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from fuzzywuzzy import process
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
st.title("Stock Price Analyzer")
st.write("This tool is developed to analyze stock data, generate plots using technical indicators, and predict stock prices")

# Load the Excel sheet
company_data = pd.read_excel("G:/Practice/Python/tickers.xlsx")
company_names = company_data["Name"].tolist()

# Default company and ticker
default_company = "Amazon.com, Inc."
default_ticker = "AMZN"

st.sidebar.header("Enter a Company Name")
company_input = st.sidebar.text_input("Type to search for a company", value=default_company)

if company_input:
    best_matches = process.extractBests(company_input, company_names, score_cutoff=70, limit=5)
    suggested_companies = [match[0] for match in best_matches] if best_matches else [default_company]
    selected_company = suggested_companies[0]
    selected_ticker = company_data.loc[company_data["Name"] == selected_company, "Ticker"].values[0]
else:
    selected_company = default_company
    selected_ticker = default_ticker

selected_company = st.sidebar.selectbox("Select a Company Name", suggested_companies, index=0 if company_input else -1)

years = st.sidebar.slider("Select Number of years of Historical Data", min_value=1, max_value=10, value=5)
show_moving_average = st.sidebar.checkbox("50 Moving Average", value=True)
years_prediction = st.sidebar.slider("Select Number of years to predict", min_value=2, max_value=10, value=5)

enable_comparison = st.sidebar.checkbox("Compare with Another Company")

if enable_comparison:
    st.sidebar.header("Compare with Another Company")
    compare_company_input = st.sidebar.text_input("Type to search for another company", value="Microsoft")
    compare_best_matches = process.extractBests(compare_company_input, company_names, score_cutoff=70, limit=5)
    compare_suggested_companies = [match[0] for match in compare_best_matches] if compare_best_matches else ["Microsoft"]
    compare_company = compare_suggested_companies[0]
    compare_ticker = company_data.loc[company_data["Name"] == compare_company, "Ticker"].values[0]

def get_stock_data(ticker_symbol, company_name, years):
    try:
        end = pd.to_datetime('today').strftime("%Y-%m-%d")
        start = (pd.to_datetime('today') - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
        
        st.write(f"Data for {company_name}")
        api_key = "d12YIOoXOYojQqUAiR7bZtElI3eliJc3"  # Your FMP API key
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker_symbol}?from={start}&to={end}&apikey={api_key}"
        
        response = requests.get(url)
        data = response.json()
        
        if "historical" not in data or not data["historical"]:
            st.warning(f"No data available for {ticker_symbol} in the specified date range.")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(data["historical"])
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        
        # Select relevant columns (FMP provides split-adjusted data by default)
        df = df[["open", "high", "low", "close"]]
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})

        if not {'Open', 'Close', 'High', 'Low'}.issubset(df.columns):
            st.warning(f"Data for {ticker_symbol} is incomplete or unavailable.")
            return pd.DataFrame()

        yearly_data = df.resample('YE').agg({"Open": "first", "Close": "last", "High": "max", "Low": "min"})
        yearly_data.index = yearly_data.index.year.astype(str)
        
        yearly_data.rename(columns={
            "High": "52 Week High", "Low": "52 Week Low",
            "Open": "Year Open", "Close": "Year Close"
        }, inplace=True)
        
        # Rename the index to "Year"
        yearly_data.index.name = "Year"

        return yearly_data

    except Exception as e:
        st.error(f"Error fetching data for {ticker_symbol}: {e}")
        return pd.DataFrame()

def plot_stock_data(data, compare_data, company_name, compare_company_name, title, show_moving_average=True, enable_comparison=False):
    fig = px.line(data, x=data.index, y='52 Week High', title=title)
    fig.add_scatter(x=data.index, y=data['52 Week High'], mode='lines', name=f'{company_name} 52 Week High')

    if enable_comparison and compare_data is not None and not compare_data.empty:
        fig.add_scatter(x=compare_data.index, y=compare_data['52 Week High'], mode='lines', name=f'{compare_company_name} 52 Week High')

    if show_moving_average:
        sma_50 = data['Year Close'].rolling(window=50, min_periods=1).mean()
        fig.add_scatter(x=data.index, y=sma_50, mode='lines', name=f'{company_name} 50-Day Moving Avg', line=dict(dash='dash'))

        if enable_comparison and compare_data is not None:
            compare_sma_50 = compare_data['Year Close'].rolling(window=50, min_periods=1).mean()
            fig.add_scatter(x=compare_data.index, y=compare_sma_50, mode='lines', name=f'{compare_company_name} 50-Day Moving Avg', line=dict(dash='dash'))

    st.plotly_chart(fig)

def predict_stock_prices(data, company_name, years_prediction):
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    closing_prices = data['Year Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(closing_prices)

    time_step = min(60, len(scaled_data) // 2)
    if len(scaled_data) < time_step:
        st.warning(f"Not enough data for {company_name} to perform predictions.")
        return pd.DataFrame()

    X_train, y_train = [], []
    for i in range(time_step, len(scaled_data)):
        X_train.append(scaled_data[i - time_step:i, 0])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    model.fit(X_train, y_train)

    predictions = []
    last_sequence = X_train[-1]
    for _ in range(years_prediction):
        prediction = model.predict(last_sequence.reshape(1, -1))
        predictions.append(prediction[0])
        last_sequence = np.append(last_sequence[1:], prediction, axis=0)

    if predictions:
        future_data = pd.DataFrame(
            index=pd.date_range(start=f"{pd.to_datetime('today').year + 1}-01-01", periods=years_prediction, freq='Y'),
            columns=['Predicted Year Close'])
        future_data['Predicted Year Close'] = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return future_data
    else:
        return pd.DataFrame()

def plot_predicted_stock_prices(stock_data, predicted_data, company_name, years_prediction, enable_comparison=False, compare_predicted_data=None, compare_company_name=""):
    if predicted_data.empty:
        st.error(f"No predicted data available for {company_name}.")
        return

    fig = px.line(predicted_data, x=predicted_data.index, y='Predicted Year Close', labels={'Predicted Year Close': 'Predicted Stock Price'},
                  title=f"{company_name} Predicted Stock Price" if not enable_comparison else f"{company_name} vs {compare_company_name} Predicted Stock Price Comparison")
    fig.add_scatter(x=predicted_data.index, y=predicted_data['Predicted Year Close'], mode='lines', name=f'{company_name} Predicted Price')

    if enable_comparison and compare_predicted_data is not None and not compare_predicted_data.empty:
        fig.add_scatter(x=compare_predicted_data.index, y=compare_predicted_data['Predicted Year Close'], mode='lines', name=f'{compare_company_name} Predicted Price')

    st.plotly_chart(fig)
    fig.update_layout(xaxis_title="Year", yaxis_title="Stock Price", legend_title="Company")

def convert_df_to_csv(df):
    return df.to_csv(index=True)  # Include index (Year) in CSV

with st.spinner("Fetching stock data..."):
    stock_data = get_stock_data(selected_ticker, selected_company, years)
    if not stock_data.empty:
        st.write(f"{selected_company} Stock Data:")
        st.write(stock_data)
        csv = convert_df_to_csv(stock_data)
        st.download_button(label=f"Download {selected_company} Stock Data", data=csv, file_name=f"{selected_company}_stock_data.csv", mime="text/csv")

        if enable_comparison:
            compare_stock_data = get_stock_data(compare_ticker, compare_company, years)
            if not compare_stock_data.empty:
                st.write(f"{compare_company} Stock Data:")
                st.write(compare_stock_data)
                compare_csv = convert_df_to_csv(compare_stock_data)
                st.download_button(label=f"Download {compare_company} Stock Data", data=compare_csv, file_name=f"{compare_company}_stock_data.csv", mime="text/csv")
        else:
            compare_stock_data = None

        if enable_comparison:
            graph_title = f"{selected_company} vs {compare_company} 52 Week High Graph"
        else:
            graph_title = f"{selected_company} 52 Week High Graph"

        plot_stock_data(stock_data, compare_stock_data, selected_company, compare_company if enable_comparison else "", graph_title, show_moving_average, enable_comparison)

        predicted_data = predict_stock_prices(stock_data, selected_company, years_prediction)
        
        if enable_comparison and compare_stock_data is not None:
            compare_predicted_data = predict_stock_prices(compare_stock_data, compare_company, years_prediction)
        else:
            compare_predicted_data = pd.DataFrame()
        
        plot_predicted_stock_prices(stock_data, predicted_data, selected_company, years_prediction, enable_comparison, compare_predicted_data, compare_company if enable_comparison else "")

        st.write("Definitions")
        st.write("52-Week High/Low: This shows the highest and lowest prices the stock has reached in the past 52 weeks") 
        st.write("Year Open/Close: This indicates the stock's price at the beginning and end of the current calendar year.")
    else:
        st.error(f"{selected_company} is delisted, merged, or data is unavailable.")
