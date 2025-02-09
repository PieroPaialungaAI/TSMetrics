# streamlit_utils.py
import streamlit as st
import numpy as np
import pandas as pd
# Import our new plotting helpers
from plot_utils import *
from TSAnalyzer import TimeSeriesAnalyzer
import sympy

def load_data():
    """
    Generate data from a user-defined expression (with Sympy) + optional Gaussian noise.
    Returns (x_data, y_data) or (None, None) if generation fails.
    """

    st.subheader("1. Generate data from a custom expression (Sympy)")

    st.markdown("""
    **Type your function in the box below,** using **'x'** as the variable.

    **Examples**:
    - `4*x + 5*log(x) + 73*sin(x)`
    - `exp(x) - 0.5*x^2`
    - `sin(x)*cos(2*x)`

    **Sympy Functions** you can use (among others):
    - `sin(x)`, `cos(x)`, `tan(x)`
    - `log(x)` (natural log), `exp(x)` (exponential)
    - `sqrt(x)`, `abs(x)`, etc.
    """)

    # Text input for the user-defined expression
    function_string = st.text_input("Function f(x)", "4*x + 5*log(x) + 73*sin(x)")

    # Domain controls
    st.write("**Choose the domain of x:**")
    x_min = st.number_input("x start (x_min)", value=1.0, step=1.0)
    x_max = st.number_input("x end (x_max)", value=20.0, step=1.0)
    num_points = st.slider("Number of Points", 10, 2000, 200, step=10)

    # Noise slider
    noise_level = st.slider("Noise Level (standard deviation)", 0.0, 5.0, 0.0, 0.1)

    x_data = None
    y_data = None

    # Button to generate data
    if st.button("Generate Data"):
        # Create our x vector
        x_data = np.linspace(x_min, x_max, num_points)

        # Use Sympy to parse and evaluate the expression
        try:
            # 1. Define a Sympy symbol for x
            x_sym = sympy.Symbol('x', real=True)
            expr = sympy.sympify(function_string)
            func = sympy.lambdify(x_sym, expr, 'numpy')
            y_data = func(x_data)

            # 5. If the user wants noise, add Gaussian noise
            if noise_level > 0.0:
                y_data = y_data + np.random.normal(scale=noise_level, size=len(x_data))

        except sympy.SympifyError as e:
            st.error(f"Invalid expression: {e}")
            x_data, y_data = None, None
        except Exception as e:
            st.error(f"Error evaluating function: {e}")
            x_data, y_data = None, None

        # Plot if successful
        if y_data is not None:
            st.write("## Generated Time Series")
            fig, ax = plt.subplots(figsize=(8, 4))
            label = "f(x) + noise" if noise_level > 0 else "f(x)"
            ax.plot(x_data, y_data, label=label)
            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.set_title("Plot of User-Defined Function (Sympy)")
            ax.legend()
            st.pyplot(fig)

    return x_data, y_data



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
    st.subheader("5. Seasonal Decomposition (Optional)")

    if analyzer is not None:
        decompose_model = st.selectbox("Model", ["additive", "multiplicative"])
        if st.button("Perform Seasonal Decomposition"):
            result = analyzer.decompose(model=decompose_model)
            st.write("Trend, Seasonal, and Residual Components:")
            # Decomposition returns a result object, so we let our helper 
            # convert it to a figure.
            fig_decomp = plot_decomposition_result(result)
            st.pyplot(fig_decomp)
    else:
        st.info("No data available. Please upload or generate data first.")
