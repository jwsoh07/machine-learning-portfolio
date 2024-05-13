import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


def credit_card_fraud_detection():
    # Main title of the page
    st.title("Credit Card Fraud Detection with Logistic Regression")

    st.subheader("Motivation:")
    st.write(
        """
        As an aspiring professional seeking opportunities within the financial sector, I understand the critical importance of 
        maintaining trust and security in financial transactions. The prevalence of credit card fraud poses significant risks 
        not only to financial institutions but also to individual consumers.
        """
    )

    st.subheader("Introduction:")
    st.write(
        """
    In this application, we use a logistic regression model to predict fraudulent transactions based on the Kaggle Credit Card 
    Fraud Detection dataset. Logistic regression is a binary classification algorithm that models the probability of an input 
    belonging to a particular class (normal or fraudulent).

    The model uses the following mathematical equation to calculate the probability \( p \) of a transaction being fraudulent:
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

    A significant difference between the counts of class 0 (normal) and class 1 (fraudulent) indicates a class imbalance, which is common in 
    fraud detection datasets. This means that there are far more normal transactions than fraudulent ones (see bar chart below).

    Class imbalance can impact the performance of machine learning models, as they may become biased towards predicting the majority class (normal transactions) more frequently. This makes it important to account for class imbalance when training a model for fraud detection.

    In this case, 'Downsampling' technique has been used to maintain the performance of the machine learning model. Downsampling is a technique used to balance class distribution in datasets where one class is significantly larger than the other(s). 
    
    This is particularly important when working with imbalanced datasets in machine learning, such as the credit card fraud detection dataset, where the majority of transactions are normal and only a small fraction are fraudulent.
    """
    )
    # Visualize the distribution of classes (normal vs fraudulent)
    sns.countplot(x="Class", data=data)
    plt.title("Distribution of Normal and Fraudulent Transactions")
    plt.xticks([0, 1], ["Normal", "Fraudulent"])
    st.pyplot(plt)

    preprocessed_data = preprocess_data(data)
    balanced_dataset = create_balanced_dataset(preprocessed_data)
    X, y = get_features_and_target(balanced_dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=2,
    )

    model = LogisticRegression(max_iter=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)

    st.subheader("Classification Report")
    # Render the classification report DataFrame as a table
    classification_df = pd.DataFrame(classification_rep).transpose()
    st.table(classification_df)

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

    st.subheader("Results")
    # Evaluate the model using confusion matrix and classification report
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix as a heatmap
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Normal", "Predicted Fraudulent"],
        yticklabels=["Normal", "Fraudulent"],
    )
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


def preprocess_data(data):
    """
    Preprocesses the data for optimal performance when the output data is fed to the logistics regression model.

    Steps:
    # 1. Performs robust scaling to transform the 'Amount' column
    # 2. Normalise the 'Time' column
    """
    preprocessed_data = data.copy()
    preprocessed_data["Amount"] = RobustScaler().fit_transform(
        preprocessed_data["Amount"].to_numpy().reshape(-1, 1)
    )
    time = preprocessed_data["Time"]
    preprocessed_data["Time"] = (time - time.min()) / (time.max() - time.min())

    preprocessed_data = preprocessed_data.sample(frac=1, random_state=1)
    return preprocessed_data


def create_balanced_dataset(data):
    normal_transactions = data[data["Class"] == 0]
    fraudulent_transactions = data[data["Class"] == 1]
    # Downsample the majority class to match the size of the minority class
    downsampled_normal_transactions = normal_transactions.sample(
        n=len(fraudulent_transactions)
    )

    # Combine the downsampled majority class and the minority class
    balanced_dataset = pd.concat(
        [
            downsampled_normal_transactions,
            fraudulent_transactions,
        ],
        axis=0,
    )

    return balanced_dataset


def get_features_and_target(data):
    X = data.drop(columns=["Class"])
    y = data["Class"]
    return X, y
