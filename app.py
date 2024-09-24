import streamlit as st
import pandas as pd
import pickle
from PIL import Image

# Function to build the prediction model
def build_model(input_data):
    # Load saved regression model
    load_model = pickle.load(open('acetylcholinesterase_model.pkl', 'rb'))
    # Apply model to make predictions
    prediction = load_model.predict(input_data)
    
    st.header('**Prediction output**')
    prediction_output = pd.Series(prediction, name='pIC50')
    molecule_name = pd.Series(load_data[1], name='molecule_name')
    
    df = pd.concat([molecule_name, prediction_output], axis=1)
    st.write(df)
    
    # Option to download the prediction output
    st.markdown(filedownload(df), unsafe_allow_html=True)

# Improved custom CSS for better styling
st.markdown("""
    <style>
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #f0f4f7;
    }
    h1 {
        color: #0066cc;
        font-size: 38px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
        background-color: #0066cc;
        color: white;
        font-size: 18px;
        border-radius: 8px;
        padding: 10px 20px;
        margin-top: 20px;
    }
    .stButton > button:hover {
        background-color: #005bb5;
    }
    .uploadedFile {
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

# Logo image (optional)
image = Image.open('logo.png')  # Replace with your own image path
st.image(image, use_column_width=True)

# Page title
st.markdown("""
# Bioactivity Prediction App (Acetylcholinesterase)
This app allows you to predict the bioactivity towards inhibiting the `Acetylcholinesterase` enzyme, a drug target for Alzheimer's disease.
---
""")

# Sidebar with file upload
st.sidebar.header('1. Upload your CSV data')
uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'], key='uploadedFile')
st.sidebar.markdown("""
[Example input file](https://raw.githubusercontent.com/dataprofessor/bioactivity-prediction-app/main/example_acetylcholinesterase.txt)
""")

# Prediction button in the sidebar
if st.sidebar.button('Predict'):
    load_data = pd.read_table(uploaded_file, sep=' ', header=None)
    load_data.to_csv('molecule.smi', sep='\t', header=False, index=False)
    
    st.header('**Original input data**')
    st.write(load_data)
    
    with st.spinner("Calculating descriptors..."):
        desc_calc()  # Call the descriptor calculation function
    
    # Display calculated descriptors
    st.header('**Calculated molecular descriptors**')
    desc = pd.read_csv('descriptors_output.csv')
    st.write(desc)
    st.write(desc.shape)
    
    # Subset of descriptors used in the previously built model
    st.header('**Subset of descriptors from previously built models**')
    Xlist = list(pd.read_csv('descriptor_list.csv').columns)
    desc_subset = desc[Xlist]
    st.write(desc_subset)
    st.write(desc_subset.shape)
    
    # Apply trained model to make prediction on query compounds
    build_model(desc_subset)
else:
    st.info('Upload input data in the sidebar to start!')

# Optional: Add download functionality for results
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Convert to base64
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download CSV File</a>'
    return href
