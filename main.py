import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from constants import * 

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
    st.title("📈 Time Series Analyzer: All your Time Series tool in one Web App!")
    st.write("Crafted with ❤️ by **Piero Paialunga**")

    # 1) Data Input
    x_label, y_label = DEFAULT_X_LABEL, DEFAULT_Y_LABEL
    x_data, y_data, x_label, y_label = load_data()
    # Construct the Analyzer only if data is available
    if x_data is not None and y_data is not None:
        analyzer = TimeSeriesAnalyzer(y=y_data, x=x_data, x_label=x_label, y_label=y_label)
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

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; font-size: 12px; margin-top: 50px;'>
            🔗 Check out my <a href='https://towardsdatascience.com/author/piero-paialunga/' target='_blank'>portfolio and projects</a>!
        </div>
        """, 
        unsafe_allow_html=True
    )

# Standard Python convention to run the main function
if __name__ == "__main__":
    main()