# TSAnalyzer

TSMetrics is a Python-based toolkit designed for evaluating and visualizing time series data using multiple metrics. It provides a comprehensive set of tools to analyze time series datasets, compute standard and specific metrics, and generate insightful visualizations.

## Features

- **Time Series Analysis**: Tools to analyze and interpret time series data.
- **Standard Metrics Computation**: Calculate common metrics such as mean, median, standard deviation, etc.
- **FFT and decomposition**: Computing the Fourier Transform and signal decompositions.
- **Forecasting**: Forecasting X steps using ARIMA.
- **Streamlit Integration**: Utilities to create interactive web applications for time series analysis using Streamlit.

## Installation

To install TSMetrics, clone the repository and install the required dependencies:

```bash
git clone https://github.com/PieroPaialungaAI/TSMetrics.git
cd TSMetrics
pip install -r requirements.txt
```

## Usage

Here's a basic example of how to use TSMetrics:

```python
from TSAnalyzer import TimeSeriesAnalyzer
import numpy as np

M = 100
x = np.linspace(-np.pi,np.pi,M)
y_true = np.abs(3.4*np.sin(x))
N = 10
y_pred = np.abs(np.array([np.random.choice(np.linspace(-5.0,5.0,M))*np.sin(x) for _ in range(N)]))
ts_analyzer = TimeSeriesAnalyzer(y_true)
ts_analyzer.descriptive_statistics()
ts_analyzer.forecast()
ts_analyzer.decompose()
```

To see the results of this, refer to the `example.ipynb` notebook provided in the repository.

## Modules

- `TSAnalyzer.py`: Main class for time series analysis.
- `streamlit_utils.py`: Helpers for creating Streamlit applications.
- `main.py`: The code to run the streamlit app.
- `utils.py`: General utility functions.
- `constants.py`: A script that contains all the constants
- `example.ipynb`: An example notebook with some useful functions.
- `TSMetrics.py` and other scripts: Work in progress 🙂

## Contributing

Contributions are welcome! If you'd like to contribute to TSMetrics, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

We appreciate the contributions of the open-source community and the developers who have made this project possible.
