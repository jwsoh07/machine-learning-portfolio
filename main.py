import streamlit as st
from streamlit_option_menu import option_menu

# use cases
import time_series_analysis.moving_average
import supervised_learning.kth_nearest_neighbours
import supervised_learning.logistic_regression

with st.sidebar:
    selected = option_menu(
        "ML Projects",
        [
            "Credit Card Fraud Detection",
            "Stock Price Moving Averages",
            "Iris Flower Classification",
        ],
        icons=["credit-card", "graph-up-arrow", "flower2"],
        menu_icon="robot",
        default_index=1,
    )


if selected == "Credit Card Fraud Detection":
    supervised_learning.logistic_regression.credit_card_fraud_detection()

if selected == "Stock Price Moving Averages":
    time_series_analysis.moving_average.artificial_use_case()

if selected == "Iris Flower Classification":
    supervised_learning.kth_nearest_neighbours.iris_flower_classification()
