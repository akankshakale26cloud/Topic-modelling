import streamlit as st
import pandas as pd
from scatter_plot import * 
import pickle
from confusion_matrices import *
from accuracies import *
from prompt import *

st.set_page_config(layout="wide")

def upload_file():
    st.session_state["scatter_plot"] = pd.read_csv("t-SNE scatter with text.csv")

def load_confusion_matrices():
    """Load confusion matrices from a pickle file."""
    with open("confusion_matrices.pkl", 'rb') as file:
        st.session_state["confusion_matrices"] = pickle.load(file)

def load_pca_accuracies():
    with open("accuracies.pkl", 'rb') as file:
        st.session_state["pca_accuracies"] = pickle.load(file)


def main():
    st.markdown("""
    <h1 style="text-align: center; font-size: 36px; font-weight: bold; color: #2E3A59;">
        AG news article text classification dashboard
    </h1>
    """, unsafe_allow_html=True)

    upload_file()
    load_confusion_matrices()
    load_pca_accuracies()
    col1, col2, col3 = st.columns([1,1,1], gap="large")
    with col1:
        populate_scatter_plot()

    with col2:
        populate_confusion_matrix()
    with col3:
        populate_accuracy_plot()
        display_accuracies_table()
    
    text = st.text_input("Enter prompt: ").strip()

    if text and text.strip()!= '':
        with st.spinner("Processing..."):
            output = find_class(tokenizer, model, logit_model, text)
            if output:
                st.write(f'### Class: {output}')
    

if __name__ == "__main__":
    tokenizer, model, logit_model = load_models()

    main()
