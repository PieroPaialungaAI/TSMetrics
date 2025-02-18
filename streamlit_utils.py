# streamlit_utils.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_utils import *
from TSAnalyzer import TimeSeriesAnalyzer
from utils import symbolic_conversion
from constants import * 


def load_data():
    # Ensure session state keys exist
    if "x_data" not in st.session_state:
        st.session_state.x_data = None
    if "y_data" not in st.session_state:
        st.session_state.y_data = None
    if "x_label" not in st.session_state:
        st.session_state.x_label = None
    if "y_label" not in st.session_state:
        st.session_state.y_label = None
    st.subheader("🚀 Generate or Upload Your Time Series Data")
    st.write("""
        **Step 1: Choose how to provide your time series data.**  
        🔹 **Generate custom data:** Define a mathematical function in terms of `x`.  
        🔹 **Upload your own data:** Upload a `.csv` file with your time series.
    """)

    data_option = st.radio("How would you like to provide your time series data?", ["Generate Custom Data", "Upload CSV File"])
    plot_container = st.container()  # Persistent container for plotting

    if data_option == "Generate Custom Data":
        function_string = st.text_input("📝 Type Your Function (in terms of `x`)", DEFAULT_TS)
        x_min = st.number_input("Start of x-axis (x_min)", value = MIN_X, step = X_STEP)
        x_max = st.number_input("End of x-axis (x_max)", value = MAX_X, step = X_STEP)
        num_points = st.slider("🔢 Number of Points to Generate", MIN_POINTS, MAX_POINTS, POINTS_STEP)
        noise_level = st.slider("🌪️ Add Noise to the Data (Standard Deviation)", MIN_NOISE, MAX_NOISE, MIN_NOISE, STEP_NOISE)
        x_label = st.text_input("🛠️ Custom x-axis Label", "Time")
        y_label = st.text_input("🛠️ Custom y-axis Label", "Value")

        if st.button("✨ Generate Time Series Data"):
            try:
                x_data = np.linspace(x_min, x_max, num_points)
                y_data = symbolic_conversion(function_string, x_data)
                if noise_level > 0:
                    y_data += np.random.normal(scale=noise_level, size=len(x_data))
                
                st.session_state.x_data = x_data
                st.session_state.y_data = y_data
                st.session_state.x_label = x_label
                st.session_state.y_label = y_label
            except Exception as e:
                st.error(f"❌ Error: {e}")

    elif data_option == "Upload CSV File":
        uploaded_file = st.file_uploader("📂 Upload Your CSV File", type=["csv"])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("### 📄 Uploaded Data Preview:")
                st.dataframe(df.head())

                column_name = st.selectbox("Select the column with your time series data", df.columns)
                y_data = df[column_name].values
                x_data = np.arange(len(y_data))
                x_label = st.text_input("🛠️ Custom x-axis Label for Uploaded Data", "Index")
                y_label = st.text_input("🛠️ Custom y-axis Label for Uploaded Data", column_name)

                st.session_state.x_data = x_data
                st.session_state.y_data = y_data
                st.session_state.x_label = x_label
                st.session_state.y_label = y_label

            except Exception as e:
                st.error(f"❌ Error loading CSV file: {e}")

    if st.session_state.x_data is not None and st.session_state.y_data is not None:
        with plot_container:
            st.write("### 📊 Your Time Series Data")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(st.session_state.x_data, st.session_state.y_data, label="Time Series")
            ax.set_xlabel(st.session_state.x_label)
            ax.set_ylabel(st.session_state.y_label)
            ax.set_title("Time Series Plot")
            ax.legend()
            st.pyplot(fig)

    return st.session_state.x_data, st.session_state.y_data, st.session_state.x_label, st.session_state.y_label


def show_descriptive_statistics(analyzer):
    st.subheader("📊 Descriptive Statistics")
    if analyzer is not None:
        if st.button("🔍 Compute Statistics"):
            stats = analyzer.descriptive_statistics()
            st.write("### 📈 Statistics Summary")
            st.dataframe(stats)
            stats_fig = analyzer.plot_descriptive_statistics()
            st.pyplot(stats_fig)
            # Convert DataFrame to CSV for download
            csv = stats.to_csv(index=False).encode('utf-8')
            file_name = st.text_input("Enter file name for download:", DEFAULT_STATISTICS_NAME)
            # Add a download button
            st.download_button(
                label="📥 Download Statistics as CSV",
                data=csv,
                file_name=file_name if file_name.endswith(".csv") else file_name + ".csv",
                mime="text/csv"
            )
    else:
        st.info("ℹ️ No data available. Please generate or upload your time series first.")


def show_cleaning_options(analyzer):
    st.subheader("🧼 Noise Reduction and Smoothing")
    if analyzer is not None:
        method = st.selectbox("Select Noise Reduction Method", ["Savitzky-Golay Filter", "FFT Low-Pass Filter"])
        if method == "Savitzky-Golay Filter":
            window_length = st.slider("Window Length (must be odd)", 3, 101, 51, step=2)
            polyorder = st.slider("Polynomial Order", 1, 5, 3)
            if st.button("🛠️ Apply Savitzky-Golay Filter"):
                analyzer.clean_noise(method='savgol', window_length=window_length, polyorder=polyorder)
                st.success("✔️ Savitzky-Golay Filter Applied!")
        else:
            cutoff = st.slider("Cutoff Frequency", 0.0, 0.5, 0.1, step=0.01)
            if st.button("🚀 Apply FFT Filter"):
                analyzer.clean_noise(method='fft', cutoff=cutoff)
                st.success("✔️ FFT Filter Applied!")
    else:
        st.info("ℹ️ No data available. Please generate or upload your time series first.")


def show_cleaning(analyzer):
    st.subheader("📊 Visualize Your Time Series")

    if analyzer is not None:
        if analyzer.cleaned_y is not None and analyzer.noise is not None:
            st.write("""
            **Here’s your cleaned time series along with the detected noise component.**  
            This helps you compare the original, cleaned, and noise-separated data in one plot!
            """)

            fig = plot_cleaned_series(analyzer.x, analyzer.y, analyzer.cleaned_y, analyzer.noise)
            st.pyplot(fig)

            st.success("✔️ Visualization complete! You can return to Step 3 to try other noise-cleaning methods.")
        else:
            st.write("⚠️ No cleaned data to display yet. Please apply a noise reduction filter in **Step 3**.")
    else:
        st.info("ℹ️ No data available. Please generate or upload your time series first.")



def show_decomposition(analyzer):
    st.subheader("📉 Detrending Your Time Series")
    if analyzer is not None:
        method = st.selectbox("Choose Detrend Method", ["Constant (Remove Mean)", "Linear (Remove Best-Fit Line)", "Polynomial (Remove Polynomial Trend)"])
        degree = 2
        if "Polynomial" in method:
            degree = st.slider("Select Polynomial Degree", 2, 10, 2)
        if st.button("🎯 Perform Detrend"):
            detrended, fig = analyzer.decompose(method=method.lower().split()[0], degree=degree)
            st.pyplot(fig)
            st.success("✔️ Detrending Complete!")
    else:
        st.info("ℹ️ No data available. Please generate or upload your time series first.")


def show_spectrogram(analyzer):
    """
    Displays the FFT spectrogram of the time series data.
    """
    st.subheader("🎵 Frequency Analysis: Spectrogram")
    
    if analyzer is not None:
        # User selects parameters for the spectrogram
        window_size = st.slider("Select Window Size for FFT", 64, 1024, SPECTOGRAM_WINDOW_SIZE, step=64)
        overlap = st.slider("Select Overlap", 0, window_size - 1, SPECTOGRAM_OVERLAP, step=8)
        cmap = st.selectbox("Choose Colormap", ["inferno", "viridis", "plasma", "magma", "jet"], index=0)

        if st.button("🎛️ Compute Spectrogram"):
            f, t, Sxx = analyzer.fft_spectogram(window_size=window_size, overlap=overlap, cmap=cmap)

            st.write("### 🔍 Spectrogram Analysis")
            fig, ax = plt.subplots(figsize=(10, 5))
            cax = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap=cmap)
            fig.colorbar(cax, ax=ax, label=f'{analyzer.y_label} Intensity')

            ax.set_title("Spectrogram (Fourier Transform Heatmap)")
            ax.set_xlabel(f"{analyzer.x_label} (Natural Domain)")
            ax.set_ylabel(f"{analyzer.x_label} Frequency")

            st.pyplot(fig)
            st.success("✔️ Spectrogram Computed!")
    else:
        st.info("ℹ️ No data available. Please generate or upload your time series first.")


def show_forecasting(analyzer):
    """
    Streamlit interface for forecasting using the ARIMA model.
    """
    st.subheader("🔮 Time Series Forecasting")

    if analyzer is not None:
        # User inputs for forecast settings
        steps = st.slider("🔢 Steps to Predict", min_value=5, max_value=100, value=10, step=5)
        p = st.number_input("📊 AR Order (p)", min_value=0, max_value=5, value=1, step=1)
        d = st.number_input("🔄 Differencing Order (d)", min_value=0, max_value=2, value=1, step=1)
        q = st.number_input("🌀 MA Order (q)", min_value=0, max_value=5, value=1, step=1)
        
        if st.button("🚀 Run Forecast"):
            # Perform forecast
            forecast_df, fig = analyzer.forecast(steps=steps, order=(p, d, q))

            # Show forecast plot
            st.pyplot(fig)

            # Convert forecast DataFrame to CSV for download
            csv = forecast_df.to_csv(index=False).encode("utf-8")

            # User input for custom file name
            file_name = st.text_input("Enter file name for download:", "forecast_results.csv")

            # Add download button
            st.download_button(
                label="📥 Download Forecast Data",
                data=csv,
                file_name=file_name if file_name.endswith(".csv") else file_name + ".csv",
                mime="text/csv"
            )
    else:
        st.info("ℹ️ No data available. Please generate or upload your time series first.")



def show_footer(portfolio_url = PORTFOLIO_URL):
    """Displays a footer with a customizable link to the portfolio."""
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; font-size: 12px; margin-top: 50px;'>
            🔗 Check out my <a href='{portfolio_url}' target='_blank'>portfolio and projects</a>!
        </div>
        """, 
        unsafe_allow_html=True
    )

