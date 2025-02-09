import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from statsmodels.tsa.seasonal import seasonal_decompose
import sympy

class TimeSeriesAnalyzer:
    def __init__(self, y, x=None):
        self.y = y
        self.x = np.arange(len(y)) if x is None else x
        self.cleaned_y = None
        self.noise = None
        self.fft_result = None

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


    def decompose(self, model='additive'):
        """Perform seasonal decomposition on the time series."""
        result = seasonal_decompose(self.y, model=model, period=int(len(self.y) / 12))
        result.plot()
        plt.show()
        return result
    

    def symbolic_conversion(self):
        x_sym = sympy.Symbol('x', real=True)
        expr = sympy.sympify(function_string)
        func = sympy.lambdify(x_sym, expr, 'numpy')
        y_data = func(x_data)
