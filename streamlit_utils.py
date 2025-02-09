# streamlit_utils.py
import streamlit as st
import numpy as np
import pandas as pd
# Import our new plotting helpers
from plot_utils import *
from TSAnalyzer import TimeSeriesAnalyzer
import sympy
import streamlit as st
import sympy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure session state keys exist
if "x_data" not in st.session_state:
    st.session_state.x_data = None
if "y_data" not in st.session_state:
    st.session_state.y_data = None

def load_data():
    st.subheader("1. Generate data from a custom expression (Sympy)")

    # ... domain selection, function input, etc. ...
    function_string = st.text_input("Function f(x)", "4*x + 5*log(x) + 73*sin(x)")
    x_min = st.number_input("x start (x_min)", value=1.0, step=1.0)
    x_max = st.number_input("x end (x_max)", value=20.0, step=1.0)
    num_points = st.slider("Number of Points", 10, 2000, 200, step=10)
    noise_level = st.slider("Noise Level (standard deviation)", 0.0, 5.0, 0.0, 0.1)

    # If user clicks "Generate Data"
    if st.button("Generate Data"):
        import sympy
        x_data = np.linspace(x_min, x_max, num_points)
        
        try:
            x_sym = sympy.Symbol('x', real=True)
            expr = sympy.sympify(function_string)
            func = sympy.lambdify(x_sym, expr, 'numpy')
            y_data = func(x_data)
            if noise_level > 0.0:
                y_data += np.random.normal(scale=noise_level, size=len(x_data))

            # Store into session state
            st.session_state.x_data = x_data
            st.session_state.y_data = y_data

            # Plot
            st.write("## Generated Time Series")
            fig, ax = plt.subplots(figsize=(8, 4))
            label = "f(x) + noise" if noise_level > 0 else "f(x)"
            ax.plot(x_data, y_data, label=label)
            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.set_title("Plot of User-Defined Function (Sympy)")
            ax.legend()
            st.pyplot(fig)

        except sympy.SympifyError as e:
            st.error(f"Invalid expression: {e}")
        except Exception as e:
            st.error(f"Error evaluating function: {e}")

    # At the end, return whatever is in session state
    return st.session_state.x_data, st.session_state.y_data



def show_descriptive_statistics(analyzer):
    st.subheader("2. Descriptive Statistics")

    if analyzer is not None:
        if st.button("Compute Descriptive Statistics"):
            stats = analyzer.descriptive_statistics()
            st.write("**Minimum:**", stats["min"])
            st.write("**Maximum:**", stats["max"])
            st.write("**Mean:**", stats["mean"])
            st.write("**Median:**", stats["median"])
            st.write("**Quartiles (25%, 50%, 75%):**", stats["quartiles"].tolist())
    else:
        st.info("No data available. Please upload or generate data first.")


def show_cleaning_options(analyzer):
    st.subheader("3. Clean Noise in the Time Series")

    if analyzer is not None:
        method = st.selectbox("Select a noise cleaning method", ["savgol", "fft"])
        
        if method == "savgol":
            window_length = st.slider("Window Length (must be odd)", 3, 101, 51, 2)
            polyorder = st.slider("Polynomial Order", 1, 5, 3)
            if window_length <= polyorder:
                st.warning("Window length must be larger than polynomial order.")
            else:
                if st.button("Apply Savitzky-Golay Filter"):
                    analyzer.clean_noise(method='savgol', 
                                         window_length=window_length, 
                                         polyorder=polyorder)
                    st.success("Savgol filter applied.")
        else:
            cutoff = st.slider("Cutoff Frequency", 0.0, 0.5, 0.1, 0.01)
            if st.button("Apply FFT Filter"):
                analyzer.clean_noise(method='fft', cutoff=cutoff)
                st.success("FFT filter applied.")
    else:
        st.info("No data available. Please upload or generate data first.")


def show_visualization(analyzer):
    st.subheader("4. Visualization")

    if analyzer is not None:
        if analyzer.cleaned_y is not None and analyzer.noise is not None:
            st.write("**Cleaned Time Series and Noise**")
            fig = plot_cleaned_series(analyzer.x, analyzer.y, analyzer.cleaned_y, analyzer.noise)
            st.pyplot(fig)
        else:
            st.write("No cleaned data to display yet. Use the filters in step 3.")
    else:
        st.info("No data available. Please upload or generate data first.")


def show_decomposition(analyzer):
    """
    Detrend the time series using one of three methods:
      - 'constant': remove the mean
      - 'linear': remove best-fit line
      - 'polynomial': remove best-fit polynomial of specified degree
    """
    st.subheader("5. Detrending (Optional)")

    if analyzer is not None:
        # Let the user choose how to detrend
        method = st.selectbox("Select a detrend method", ["constant", "linear", "polynomial"])

        degree = 2  # default polynomial degree
        if method == "polynomial":
            degree = st.slider("Polynomial degree", min_value=2, max_value=10, value=2)

        # Button triggers the detrending
        if st.button("Perform Detrend"):
            # The 'decompose' method in the analyzer class is renamed or repurposed
            # so it does detrending instead of seasonal_decompose.
            # e.g.: analyzer.decompose(method='linear', degree=2)
            detrended, fig = analyzer.decompose(method=method, degree=degree)
            st.pyplot(fig)
            # 'analyzer.decompose' already plots original vs detrended.
            # If you want additional text or checks here, do so.
            st.success("Detrending complete! See the plot above.")
    else:
        st.info("No data available. Please upload or generate data first.")
