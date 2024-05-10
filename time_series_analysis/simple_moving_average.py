import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def artificial_use_case():
    # Set the title of the Streamlit app
    st.title("Moving Average Visualization")

    # Generate dummy data
    np.random.seed(42)
    date_range = pd.date_range(start="2024-05-01", periods=100, freq="D")
    data = np.cumsum(np.random.randn(100))

    # Create a pandas DataFrame
    df = pd.DataFrame({"Date": date_range, "Value": data})

    # Add noise to the data to simulate real-world fluctuations
    df["Value"] += np.random.normal(0, 0.5, size=len(df))

    # User input for window size
    window_size = st.slider(
        "Select SMA Window Size", min_value=1, max_value=30, value=5
    )

    # Calculate Simple Moving Average
    df["SMA"] = df["Value"].rolling(window=window_size).mean()

    # Calculate Exponential Moving Average (EMA)
    df["EMA"] = df["Value"].ewm(span=window_size, adjust=False).mean()

    # Add toggles for SMA and EMA
    show_sma = st.checkbox("Show Simple Moving Average (SMA)", value=True)
    show_ema = st.checkbox("Show Exponential Moving Average (EMA)", value=True)

    # Plot the data
    fig, ax = plt.subplots()

    # Plot the original data
    ax.plot(df["Date"], df["Value"], label="Original Data", color="blue")

    # Plot the Simple Moving Average if toggle is on
    if show_sma:
        ax.plot(df["Date"], df["SMA"], label=f"SMA ({window_size}-day)", color="red")

    # Plot the Exponential Moving Average if toggle is on
    if show_ema:
        ax.plot(df["Date"], df["EMA"], label=f"EMA ({window_size}-day)", color="green")

    # Add labels and legend
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title("Simple Moving Average (SMA)")
    ax.legend()

    # Rotate the x-axis labels counterclockwise by 45 degrees
    plt.xticks(rotation=35, ha="right")

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Explanation for the plot
    st.subheader("Explanation:")
    st.write(
        """
    The plot above shows the original time series data (in blue), the Simple Moving Average (SMA) with the selected window size (in red), and the Exponential Moving Average (EMA) with the same window size (in green). 
    - The original data represents the value of a financial metric (e.g., stock price) over a given date range.
    - The SMA smooths out short-term fluctuations and helps identify underlying trends in the data. It is calculated by taking the average of the specified window size (in days) of the most recent data points.
    - The EMA assigns more weight to recent data points, making it more responsive to changes in the data compared to the SMA.

    By adjusting the window size using the slider, you can observe how different window sizes affect the smoothness and responsiveness of both the SMA and EMA.
    """
    )

    # Optional: Show the data table
    st.subheader("Data Table")
    st.write(df)

    # Explanation for the data table
    st.write(
        """
    This data table shows the date range, original values, and the calculated Simple Moving Average (SMA) for each date in the series. 
    The SMA column displays the calculated SMA based on the selected window size.
    """
    )

    # Optional: Show the data as a line chart using Streamlit
    st.subheader("Line Chart")
    st.line_chart(df.set_index("Date"))

    # Explanation for the line chart
    st.write(
        """
    This line chart presents the original data (in blue) and the Simple Moving Average (SMA) based on the selected window size (in red). 
    The interactive chart allows you to zoom in and explore different segments of the data.
    """
    )
