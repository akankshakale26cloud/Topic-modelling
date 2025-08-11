import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

def display_accuracies_table():
    """Display a table of accuracies."""
    data = {
        "Method": [
            "Logistic Regression with 768 features",
            "Features reduced to 50 using PCA",
            "Features reduced to 10 using PCA"
        ],
        "Accuracy": ["91.21%", "89.34%", "86.55%"]
    }

    df = pd.DataFrame(data)

    # st.subheader("Accuracies Table")
    st.markdown("<h2 style='text-align: center; font-size: 24px;'>Accuracies Table</h2>", unsafe_allow_html=True)
    st.table(df)

def populate_accuracy_plot():
    st.markdown("<h2 style='text-align: center; font-size: 24px;'>Dimensionality Reduction using PCA</h2>", unsafe_allow_html=True)
    # List of accuracies
    accuracies = st.session_state["pca_accuracies"]

    # List of feature numbers (ranging from 10 to 768)
    features = list(range(10, len(accuracies) * 10 + 1, 10))

    # Plot the accuracies against the number of features
    plt.figure(figsize=(8, 5))
    plt.plot(features, accuracies, marker='o', color='b', linestyle='-', linewidth=2, markersize=2)

    # Add labels and title
    plt.title('Logistic Regression Accuracy vs Number of Features')
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.gca().invert_xaxis()

    # Display the plot in Streamlit
    st.pyplot(plt)
