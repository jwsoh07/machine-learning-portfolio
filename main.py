import streamlit as st
import streamlit_antd_components as sac

# use cases
import supervised_learning.kth_nearest_neighbours

# Sidebar
with st.sidebar:
    st.subheader("Types of Machine Learning")
    st.session_state.option = sac.menu(
        [
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
                        "sadd",
                    ),
                ],
            ),
            sac.MenuItem(
                "Reinforcement Learning",
                children=[
                    sac.MenuItem(
                        "sadd",
                    ),
                ],
            ),
        ]
    )

supervised_learning.kth_nearest_neighbours.iris_flower_classification()
