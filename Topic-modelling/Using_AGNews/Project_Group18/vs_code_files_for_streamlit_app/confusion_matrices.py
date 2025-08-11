import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


def calculate_accuracy(confusion_matrix):
    """Calculate accuracy from a confusion matrix."""
    correct_predictions = np.trace(confusion_matrix)
    total_predictions = np.sum(confusion_matrix)
    return correct_predictions / total_predictions

def plot_confusion_matrix(confusion_matrix, model_name):
    """Plot the confusion matrix using Seaborn."""
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    st.pyplot(plt)

def populate_confusion_matrix():
    confusion_matrices = st.session_state["confusion_matrices"]

    if confusion_matrices is not None:
        expected_keys = {'euclidean_nn', 'cosine_nn', 'logit_reg'}

        if not expected_keys.issubset(confusion_matrices.keys()):
            st.error(f"The pickle file must contain confusion matrices for: {', '.join(expected_keys)}")
            return

        # st.text("## Confusion Matrices")
        st.markdown("<h2 style='text-align: center; font-size: 24px;'>Confusion Matrices</h2>", unsafe_allow_html=True)

        # Dropdown
        model_choice = st.selectbox("Choose a model to view its confusion matrix:",
                                    options=['euclidean_nn', 'cosine_nn', 'logit_reg'],
                                    index=0)

        # Get the confusion matrix
        confusion_matrix = confusion_matrices[model_choice]

        plot_confusion_matrix(confusion_matrix, model_choice)

        # Calculate and display accuracy
        accuracy = calculate_accuracy(confusion_matrix)
        st.write(f"##### Accuracy: {accuracy:.2%}")
    else:
        st.info("Please upload a .pkl file to proceed.")