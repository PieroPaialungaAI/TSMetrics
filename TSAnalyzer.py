import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, detrend
from statsmodels.tsa.seasonal import seasonal_decompose
import sympy
import pandas as pd 

class TimeSeriesAnalyzer:
    def __init__(self, y, x = None, x_label = 'Time', y_label ='Value'):
        self.y = y
        self.x = np.arange(len(y)) if x is None else x
        self.cleaned_y = None
        self.noise = None
        self.fft_result = None
        self.x_label = x_label
        self.y_label = y_label
        self.y = self.filter_y()
         

    def filter_y(self):
        if np.any(np.isnan(self.y)) is False:
            return self.y
        else:
            return np.nan_to_num(self.y,nan = np.nanmean(self.y))



    def perform_fft(self):
        """Compute and return the Fast Fourier Transform."""
        self.fft_result = np.fft.fft(self.y)
        return self.fft_result

    def descriptive_statistics(self):
        """Compute min, max, mean, median, and quartiles."""
        return {
            "min": np.min(self.y),
            "max": np.max(self.y),
            "mean": np.mean(self.y),
            "median": np.median(self.y),
            "quartiles": np.percentile(self.y, [25, 50, 75])
        }

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
    

