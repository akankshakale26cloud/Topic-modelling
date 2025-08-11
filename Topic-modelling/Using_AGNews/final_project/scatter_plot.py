import streamlit as st
import plotly.express as px
import pandas as pd

def populate_scatter_plot():    
    st.markdown("<h2 style='text-align: center; font-size: 24px;'>T-SNE Plot of Embeddings data</h2>", unsafe_allow_html=True)

    label_map = {0: "world", 1: "sports", 2: "business", 3: "sci/tech"}

    # File uploader for CSV file
    df = st.session_state["scatter_plot"]

    if df is not None:
        required_columns = {'tsne_x', 'tsne_y', 'label_x', 'text'}
        if not required_columns.issubset(df.columns):
            st.error(f"The uploaded file must contain the following columns: {', '.join(required_columns)}")
            return

        df['class'] = df['label_x'].map(label_map)

        color_map = {
        "world": "cyan", 
        "sports": "yellow", 
        "business": "red",  
        "sci/tech": "purple" 
    }

        # Scatter plot using Plotly
        fig = px.scatter(
            df,
            x='tsne_x',
            y='tsne_y',
            color='class',
            color_discrete_map=color_map,
            hover_data={'text': True, 'tsne_x': False, 'tsne_y': False, 'label_x': False, 'class': False},
            title="Scatter Plot",
            labels={"class": "Category"},
            width = 600,
            height= 500
        )

        fig.update_layout(
            legend=dict(
                orientation="h",  
                yanchor="bottom", 
                y=-0.2,            
                xanchor="center", 
                x=0.5             
            )
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("t-SNE scatter with text.csv not found.")