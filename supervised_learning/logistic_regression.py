import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


def fraud_detection():
    # Main title of the page
    st.title("Fraud Detection with Logistic Regression")

    st.subheader("Introduction:")
    st.write(
        """
    In this application, we use a logistic regression model to predict fraudulent transactions based on the Kaggle Credit Card Fraud Detection dataset. Logistic regression is a binary classification algorithm that models the probability of an input belonging to a particular class (normal or fraudulent).

    The model uses the following equation to calculate the probability \( p \) of a transaction being fraudulent:
    """
    )

    # Display the logistic regression equation using st.latex
    st.latex(
        r"""
    p = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}
    """
    )

    # Explanation of the equation
    st.write(
        """
    Where:
    - \( p \) is the probability of a transaction being fraudulent.
    - \( \sigma \) is the logistic (or sigmoid) function, which maps the linear combination \( z = w^T x + b \) to a probability between 0 and 1.
    - \( w^T \) is the transpose of the weight vector, representing the model's learned weights for each feature in the input data \( x \).
    - \( x \) is the input feature vector.
    - \( b \) is the bias term, representing the model's learned bias.

    The model uses a decision threshold (e.g., 0.5) to classify transactions as normal or fraudulent based on the calculated probability.
    """
    )

    # Load the Credit Card Fraud Detection dataset from Kaggle
    # URL to download the dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

    # Load the dataset into a DataFrame (adjust the file path as needed)
    data = pd.read_csv("datasets/creditcard.csv")

    # Explanation of the dataset
    st.subheader("Dataset Explanation:")
    st.write(
        """
    The dataset contains information about credit card transactions, including various features such as transaction amount, time, and other characteristics that may indicate fraudulent activity. Each row represents a transaction, and the target variable 'Class' indicates whether the transaction is normal (0) or fraudulent (1).

    For example:
    - The 'Time' column represents the elapsed time since the first transaction in the dataset.
    - The 'Amount' column represents the transaction amount.
    - Other columns represent various features derived from the transaction data.

    This data is commonly used to train models for fraud detection, enabling them to learn patterns of normal and suspicious transactions.
    """
    )

    # Display first few rows of the dataset
    st.write("Data Preview:", data.head())

    st.subheader("Class Distribution Explanation:")
    st.write(
        """
    The data distribution shown above displays the counts of transactions labeled as normal (class 0) and fraudulent (class 1) in the dataset.

    A significant difference between the counts of class 0 (normal) and class 1 (fraudulent) indicates a class imbalance, which is common in fraud detection datasets. This means that there are far more normal transactions than fraudulent ones.

    Class imbalance can impact the performance of machine learning models, as they may become biased towards predicting the majority class (normal transactions) more frequently. This makes it important to account for class imbalance when training a model for fraud detection.
    """
    )

    # Check data distribution (class imbalance)
    st.write(data["Class"].value_counts())

    # Visualize the distribution of classes (normal vs fraudulent)
    sns.countplot(x="Class", data=data)
    plt.title("Distribution of Normal and Fraudulent Transactions")
    # Customize the x-axis labels to display "Normal" and "Fraudulent"
    plt.xticks([0, 1], ["Normal", "Fraudulent"])
    st.pyplot(plt)

    # Define features (X) and target (y)
    X = data.drop(columns=["Class"])
    y = data["Class"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize the logistic regression model
    model = LogisticRegression(max_iter=1000)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    classification_rep = classification_report(y_test, y_pred)

    st.subheader("Classification Report")
    st.write(classification_rep)
    st.write("Explanation:")
    st.write(
        """
    The classification report provides an overview of the model's performance on the test data for each class (normal and fraudulent transactions). It includes the following metrics:

    - **Precision**: Precision is the ratio of true positives (correctly predicted fraudulent transactions) to the total number of positive predictions (both true positives and false positives). A high precision means that when the model predicts a transaction as fraudulent, it is likely to be correct.

    - **Recall (Sensitivity)**: Recall measures the proportion of actual fraudulent transactions that were correctly identified by the model. It is calculated as the ratio of true positives to the total number of actual positives (both true positives and false negatives). A high recall indicates that the model is effective at identifying most of the fraudulent transactions.

    - **F1-Score**: The F1-score is the harmonic mean of precision and recall. It provides a balanced measure of the model's performance, particularly when there is class imbalance. A high F1-score indicates a good balance between precision and recall.

    - **Support**: Support is the number of actual instances of each class in the test data. It provides context for the other metrics and can help assess how well the model performs given the distribution of classes.

    When interpreting the classification report, consider the balance between precision and recall. Depending on the use case, you may prioritize one over the other. For example, in fraud detection, a high recall may be more important to ensure that as many fraudulent transactions as possible are identified, even if some normal transactions are mistakenly flagged as fraudulent.
    """
    )

    st.subheader("Confusion Matrix:")
    # Evaluate the model using confusion matrix and classification report
    conf_matrix = confusion_matrix(y_test, y_pred)
    st.write(conf_matrix)

    # Plot the confusion matrix as a heatmap
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix Heatmap")
    st.pyplot(plt)

    # Explanation of the confusion matrix heatmap
    st.write(
        """
    The heatmap above shows the confusion matrix, which summarizes the model's performance on the test data:

    - The top left cell represents the number of true negatives (correctly predicted normal transactions).
    - The top right cell represents the number of false positives (incorrectly predicted fraudulent transactions).
    - The bottom left cell represents the number of false negatives (incorrectly predicted normal transactions).
    - The bottom right cell represents the number of true positives (correctly predicted fraudulent transactions).

    A well-performing model will have high values in the top left and bottom right cells, indicating accurate predictions for both normal and fraudulent transactions.
    """
    )
