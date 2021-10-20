import streamlit as st
import pandas as pd
import numpy as np

st.title('DICOM heart, lungs and trachea segmentation')

uploaded_file = st.file_uploader("Choose a DICOM file")

