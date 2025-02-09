import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from TSAnalyzer import TimeSeriesAnalyzer
from streamlit_utils import * 

###############################################################################
# Streamlit Config
###############################################################################
st.set_page_config(page_title="Time Series Analyzer", layout="centered")



def main():
    """
    Orchestrate all steps in a clean, modular way.
    """
    st.title("Time Series Analyzer (Modular Version)")

    # 1) Data Input
    
    x_data, y_data = load_data()
    # Construct the Analyzer only if data is available
    if x_data is not None and y_data is not None:
        analyzer = TimeSeriesAnalyzer(y=y_data, x=x_data)
    else:
        analyzer = None


    
    # 2) Descriptive Statistics
    show_descriptive_statistics(analyzer)
    # 3) Noise Cleaning
    show_cleaning_options(analyzer)
    # 4) Visualizations
    show_visualization(analyzer)
    # 5) Seasonal Decomposition
    show_decomposition(analyzer)


# Standard Python convention to run the main function
if __name__ == "__main__":
    main()