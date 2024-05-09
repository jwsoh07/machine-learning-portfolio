import streamlit as st

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# build a model for classifying iris flowers into one of the three species
def iris_flower_classification():
    iris = load_iris()
    X = iris.data  # Features
    y = iris.target  # Labels
    feature_names = iris.feature_names
    target_names = iris.target_names

    # Reason for dataset splitting
    # 1. Evaluation of Model Performance:
    # Using a test set that the model has not seen during training allows you to
    # objectively evaluate the model's performance. If you train a model on the
    # entire dataset and then test it on the same data, you may end up overestimating
    # the model's performance.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # What standardization does and why it is beneficial
    # 1. Mean Centering: Standardization centers the data around zero by subtracting the mean
    # of each feature. This helps in aligning all features to a common scale and improves
    # numerical stability.
    # 2. caling: Standardization scales the data so that each feature has a standard deviation
    # of one. This puts all features on a similar scale, making them comparable.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create a K-Nearest Neighbors (KNN) classifier with k=3
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Streamlit Application
    st.title("Iris Classification with K-Nearest Neighbors")
    st.header("Model Performance")
    st.markdown(
        f"""
    **Accuracy**: The K-Nearest Neighbors (KNN) model achieved an accuracy of **{accuracy * 100:.2f}%** on the test data!
    This high accuracy shows that the model performs extremely well in classifying the different species of iris flowers.
    """
    )

    # Confusion matrix
    st.subheader("Confusion Matrix")

    # Create a confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    correct_predictions = np.trace(cm)
    total_samples = len(y_test)

    # Plot the confusion matrix
    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Add text explanation about the confusion matrix, including correct predictions
    st.write(
        f"""
    The confusion matrix shows how many samples from each class (species of iris flower) were correctly or incorrectly classified.
    The rows represent the actual classes, while the columns represent the predicted classes.
    The diagonal values indicate the number of correct predictions for each class.

    The model made **{correct_predictions} correct predictions** out of a total of **{total_samples}** test samples.
    """
    )

    st.subheader("Classification Report")
    # Display classification report
    report = classification_report(y_test, y_pred, target_names=target_names)
    st.text(report)

    st.write(
        """
    The classification report provides detailed information about the performance of the model for each class (species).

    - **Precision**: Precision measures the proportion of true positive predictions out of all positive predictions made by the model. In other words, it shows how many of the predicted positive samples were actually correct.

    - **Recall**: Recall measures the proportion of true positive predictions out of all actual positive samples. It indicates how well the model can identify all relevant samples.

    - **F1-Score**: The F1-score is the harmonic mean of precision and recall. It provides a balanced measure of a model's performance, especially when the classes are imbalanced.

    - **Support**: Support is the number of actual samples for each class in the test set.

    The current scores for precision, recall, and F1-score indicate how well the model performs in identifying each species of iris flowers. High scores for these indicators reflect a well-performing model.
    """
    )

    # Scatter plot of the first two features
    st.subheader("Feature Distribution Scatter Plot")
    fig, ax = plt.subplots()
    for target, color, label in zip(range(3), ["r", "g", "b"], target_names):
        idx = y == target
        ax.scatter(X[idx, 0], X[idx, 1], c=color, label=label)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.legend()
    st.pyplot(fig)

    st.write(
        """
    This scatter plot shows the distribution of the first two features (sepal length and sepal width) for the three species of iris flowers.
    Each point represents a sample, colored by the species it belongs to.
    This visualization helps us understand how the features are distributed across the different species.
    """
    )
