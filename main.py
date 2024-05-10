import streamlit as st
import streamlit_antd_components as sac

# use cases
import time_series_analysis.simple_moving_average
import supervised_learning.kth_nearest_neighbours

# Sidebar
with st.sidebar:
    st.subheader("Types of Machine Learning")
    st.session_state.selected_model = sac.menu(
        [
            sac.MenuItem(
                "Time Series Analysis",
                children=[
                    sac.MenuItem(
                        "Moving Average",
                    ),
                ],
            ),
            sac.MenuItem(
                "Supervised Learning",
                children=[
                    sac.MenuItem(
                        "k-Nearest Neighbors (k-NN)",
                    ),
                ],
            ),
            sac.MenuItem(
                "Unsupervised Learning",
                children=[
                    sac.MenuItem(
                        "Placeholder",
                    ),
                ],
            ),
            sac.MenuItem(
                "Reinforcement Learning",
                children=[
                    sac.MenuItem(
                        "Placeholder",
                    ),
                ],
            ),
        ]
    )

if st.session_state.selected_model == "Moving Average":
    time_series_analysis.simple_moving_average.artificial_use_case()

if st.session_state.selected_model == "k-Nearest Neighbors (k-NN)":
    supervised_learning.kth_nearest_neighbours.iris_flower_classification()
