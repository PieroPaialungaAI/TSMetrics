import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, detrend
from statsmodels.tsa.seasonal import seasonal_decompose
import sympy
import pandas as pd 
from scipy.signal import spectrogram
from constants import * 
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA


class TimeSeriesAnalyzer:
    def __init__(self, y, x = None, x_label = 'Time', y_label ='Value'):
        self.y = y
        self.x = np.arange(len(y)) if x is None else x
        self.cleaned_y = None
        self.noise = None
        self.fft_result = None
        self.x_label = x_label
        self.y_label = y_label
        self.y, self.has_nan = self.filter_y()

        
    def filter_y(self):
        if np.any(np.isnan(self.y)) is False:
            return self.y, False
        else:
            return np.nan_to_num(self.y,nan = np.nanmean(self.y)), True


    def perform_fft(self):
        """Compute and return the Fast Fourier Transform."""
        self.fft_result = np.fft.fft(self.y)
        return self.fft_result


    def descriptive_statistics(self):
        """Compute min, max, mean, median, and quartiles."""
        ans =  {
            "min": np.min(self.y),
            "max": np.max(self.y),
            "mean": np.mean(self.y),
            "median": np.median(self.y),
            "quartiles": np.percentile(self.y, QUARTILES)
        }
        stats_df = pd.DataFrame({"Statistic": STATISTICS_DF_COLUMNS,
                "Value": [ans["min"], ans["max"], ans["mean"], ans["median"], 
                          ans["quartiles"][0], ans["quartiles"][1], ans["quartiles"][2]]
            })
        self.stats_df = stats_df
        return stats_df
    
    
    def plot_descriptive_statistics(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x="Statistic", y="Value", data=self.stats_df, ax=ax, palette="viridis")
        ax.set_title("Descriptive Statistics")
        ax.set_xlabel("Statistic")
        ax.set_ylabel("Value")
        plt.xticks(rotation=45)  # Rotate labels for better readability
        return fig


    def clean_noise(self, method='savgol', **kwargs):
        """Clean the noise in the time series using different filtering methods."""
        if method == 'savgol':
            window_length = kwargs.get('window_length', 51)
            polyorder = kwargs.get('polyorder', 3)
            cleaned_y = savgol_filter(self.y, window_length, polyorder)
            noise = self.y - cleaned_y
        elif method == 'fft':
            cleaned_y, noise = self._fft_filter(**kwargs)
        else:
            raise ValueError(f"Method '{method}' not recognized.")
        
        self.cleaned_y, self.noise = cleaned_y, noise
        return cleaned_y, noise


    def _fft_filter(self, cutoff=0.1):
        """Apply a low-pass filter using FFT to remove high-frequency noise."""
        fft_coeffs = np.fft.fft(self.y)
        frequencies = np.fft.fftfreq(len(self.y))
        fft_coeffs[np.abs(frequencies) > cutoff] = 0  # Zero out high frequencies
        cleaned_y = np.fft.ifft(fft_coeffs).real
        noise = self.y - cleaned_y
        return cleaned_y, noise


    def decompose(self, method='linear', degree=2):
        """
        Detrend the time series with three possible options:
          - 'constant': remove the mean
          - 'linear': remove best-fit line
          - 'polynomial': remove best-fit polynomial of given degree
        """
        if method not in ['constant', 'linear', 'polynomial']:
            raise ValueError("method must be one of {'constant','linear','polynomial'}")

        if method in ['constant', 'linear']:
            # Use scipy.signal.detrend
            detrended = detrend(self.y, type=method)
        else:
            # Polynomial detrending
            coeffs = np.polyfit(self.x, self.y, degree)
            trend = np.polyval(coeffs, self.x)
            detrended = self.y - trend

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.x, self.y, label="Original", alpha=0.7)
        ax.plot(self.x, detrended, label=f"Detrended ({method})", alpha=0.7)
        ax.set_title(f"Signal Detrend - Method: {method}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)
        # Instead of plt.show(), return the array and the figure
        return detrended, fig
    

    def fft_spectogram(self, window_size = SPECTOGRAM_WINDOW_SIZE, overlap= SPECTOGRAM_OVERLAP, cmap= SPECTOGRAM_CMAP):
        """
        Computes and plots the spectrogram of a given signal.
        
        Parameters:
        - signal: 1D NumPy array of the time-series signal.
        - sampling_rate: Sampling rate of the signal (Hz).
        - window_size: Size of the FFT window (default=256).
        - overlap: Overlap between windows (default=128).
        - cmap: Colormap for the heatmap.
        
        Returns:
        - f: Frequencies array.
        - t: Time bins array.
        - Sxx: Spectrogram matrix.
        """
        # Compute the spectrogram using STFT
        signal = self.y
        sampling_rate = 1/(self.x[1]-self.x[0])

        f, t, Sxx = spectrogram(signal, fs=sampling_rate, nperseg=window_size, noverlap=overlap)

        # Convert to log scale for better visualization
        Sxx_log = 10 * np.log10(Sxx + 1e-10)  # Adding a small constant to avoid log(0)

        # Plot the heatmap
        fig, ax = plt.subplots(figsize=(10, 5))
        cax = ax.pcolormesh(t, f, Sxx_log, shading='gouraud', cmap=cmap)
        fig.colorbar(cax, ax=ax, label=f'{self.y_label} Intensity')

        # Labels and title
        ax.set_title("Spectrogram (Fourier Transform Heatmap)")
        ax.set_xlabel(f"{self.x_label} natural domain")
        ax.set_ylabel(f"{self.x_label} frequency")
        
        plt.show()
        
        return f, t, Sxx


    def forecast(self, steps=10, order=(1,1,1), plot=True):
        """
        Forecast future values using ARIMA model.
        
        Parameters:
        - steps (int): Number of future steps to predict.
        - order (tuple): ARIMA order (p, d, q). Default is (1,1,1).
        - plot (bool): Whether to plot the forecast.
        
        Returns:
        - forecast_df (DataFrame): DataFrame with forecast values and confidence intervals.
        """
        # Fit ARIMA model
        model = ARIMA(self.y, order=order)
        model_fit = model.fit()

        # Generate forecast
        forecast_result = model_fit.get_forecast(steps=steps)
        forecast_values = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()  # ✅ Fix: Convert to NumPy array before indexing

        # Create DataFrame for forecasted values
        forecast_index = np.arange(len(self.y), len(self.y) + steps)
        forecast_df = pd.DataFrame({
            "Time": forecast_index,
            "Forecast": forecast_values,  # ✅ Fix: Ensure forecast values are NumPy arrays
            "Lower Bound": conf_int[:, 0],  # ✅ Now works correctly
            "Upper Bound": conf_int[:, 1]
        })

        # Plot if requested
        if plot:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(self.x[:-steps], self.y[:-steps], label="Original Data", marker='o', linestyle='-', alpha=0.7, color='navy')
            ax.plot(self.x[-steps:], self.y[-steps:], marker='o', linestyle='-', alpha=0.1, color='navy')
            ax.plot(self.x[-steps:], forecast_values, label="Forecast", marker='o', linestyle='--', color='r')
            ax.fill_between(self.x[-steps:], conf_int[:,0], conf_int[:,1], color='r', alpha=0.2)
            
            ax.set_title("Time Series Forecast (ARIMA)")
            ax.set_xlabel(self.x_label)
            ax.set_ylabel(self.y_label)
            ax.legend()
            ax.grid(True)
            plt.show()

        return forecast_df, fig