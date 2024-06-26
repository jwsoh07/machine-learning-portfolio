import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


def artificial_use_case():
    st.title("Moving Average Visualization with Real Financial Data")

    # Table of the top 10 Singaporean companies with their names and ticker symbols
    ten_singapore_companies = pd.DataFrame(
        {
            "Company Name": [
                "DBS Group Holdings Ltd",
                "Oversea-Chinese Banking Corporation Ltd",
                "United Overseas Bank Ltd",
                "Singapore Telecommunications Limited",
                "CapitaLand Integrated Commercial Trust",
                "Singapore Exchange Limited",
                "Singapore Airlines Limited",
                "Keppel Corporation Limited",
                "CapitaLand Group",
                "Wilmar International Limited",
            ],
            "Ticker Symbol": [
                "D05.SI",
                "O39.SI",
                "U11.SI",
                "Z74.SI",
                "C38U.SI",
                "S68.SI",
                "C6L.SI",
                "BN4.SI",
                "C31.SI",
                "F34.SI",
            ],
        }
    )

    # Display the table of the top 10 Singaporean companies with their names and ticker symbols
    st.subheader("Some Companies of Singapore Origin")
    st.table(ten_singapore_companies)

    # Input for the ticker symbol and the date range
    ticker_symbol = st.text_input("Enter Stock Ticker Symbol", value="AAPL")

    # Retrieve company information using yfinance
    company_info = yf.Ticker(ticker_symbol)
    company_name = company_info.info.get("longName", "Unknown Company")

    # Display the company name
    st.subheader(f"Company: {company_name}")

    start_date = st.date_input("Start Date", value=pd.Timestamp("2023-01-01"))
    end_date = st.date_input("End Date", value=pd.Timestamp("2024-01-01"))

    # Download data using yfinance
    data = yf.download(ticker_symbol, start=start_date, end=end_date)

    # User input for window size
    window_size = st.slider(
        "Select Window Size for Moving Averages", min_value=1, max_value=30, value=5
    )

    # Calculate Simple Moving Average (SMA)
    data["SMA"] = data["Close"].rolling(window=window_size).mean()

    # Calculate Exponential Moving Average (EMA)
    data["EMA"] = data["Close"].ewm(span=window_size, adjust=False).mean()

    # Add toggles for SMA and EMA
    show_sma = st.checkbox("Show Simple Moving Average (SMA)", value=True)
    show_ema = st.checkbox("Show Exponential Moving Average (EMA)", value=True)

    # Plot the data
    fig, ax = plt.subplots()

    # Plot the original data
    ax.plot(
        data.index,
        data["Close"],
        label="Original Data (Close Price)",
        color="blue",
    )

    # Plot the Simple Moving Average if toggle is on
    if show_sma:
        ax.plot(data.index, data["SMA"], label=f"SMA ({window_size}-day)", color="red")

    # Plot the Exponential Moving Average if toggle is on
    if show_ema:
        ax.plot(
            data.index, data["EMA"], label=f"EMA ({window_size}-day)", color="green"
        )

    # Add labels and legend
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.set_title(f"{company_name} ({ticker_symbol}) Moving Averages")
    ax.legend()

    # Rotate the x-axis labels counterclockwise by 45 degrees
    plt.xticks(rotation=45, ha="right")

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Explanation for the plot
    st.subheader("Explanation:")
    st.write(
        f"The plot above shows the original close price data for {company_name} (in blue), with optional Simple Moving Average (SMA) in red and Exponential Moving Average (EMA) in green based on the selected window size."
    )

    # Explanation of SMA and EMA
    st.write(
        """
    - **Simple Moving Average (SMA)** is a method of smoothing out price data by calculating the average of a stock's closing prices over a specified number of days. SMA helps identify trends by reducing short-term fluctuations and providing a clearer picture of the underlying trend in the stock's price.
    - **Exponential Moving Average (EMA)** is another method of smoothing out price data, but it gives more weight to recent prices, making it more responsive to recent changes in the data. EMA can be useful for identifying trends more quickly compared to SMA.

    Both SMA and EMA are plotted on the chart to give you a visual representation of the stock's performance over time. By toggling the checkboxes above the plot, you can choose whether to display SMA and EMA and adjust the window size to see how different settings impact the trends shown on the chart.
    """
    )

    # Optional: Show the data table
    st.subheader("Data Table")
    st.write(data)

    # Explanation for the data table
    st.write(
        f"This data table shows the date range, original close prices, Simple Moving Average (SMA), and Exponential Moving Average (EMA) for {company_name}."
    )
    st.write(
        """
    The SMA and EMA columns display the calculated averages based on the selected window size.
    """
    )
