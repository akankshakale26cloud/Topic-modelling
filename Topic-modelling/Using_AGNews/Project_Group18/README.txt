This file is for the basic instructions in navigating the codes that are needed to be run in order for smooth execution flow

We have ipynb file (python_major_proj_final_team18) which we executed on google colab. First we mount the google drive and do all the imports and execute the cells for the database creation and saving a few files for the visualization. We used T4 GPU to parallely process a batch of 512 records for fetching the embeddings of the texts. Once the embeddings are fetched and written to the embeddings_data table, other cells must run without errors.

Next I have also included a folder (vs_code_files_for_streamlit_app) which has the .py files and other pickle files which can help us run the streamlit app locally.