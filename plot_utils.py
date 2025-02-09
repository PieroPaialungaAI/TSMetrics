import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_comparison(y_target, y_pred, closest_curves, metric='mae'):
    """Plot the target curve in red and the closest comparison curves in grey based on a given metric."""
    plt.figure(figsize=(12, 6))
    x = np.arange(len(y_target))
    for curve in closest_curves:
        plt.plot(x, curve, color='grey', alpha=1.0, linewidth=0.8)
    plt.plot(x, y_target, color='red', linewidth=2, label='Target Curve')
    plt.title(f'Comparison of Target Curve and Closest Curves (Metric: {metric})')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid()
    plt.legend()
    plt.show()

def plot_best_match(y_target, best_match, metric='mae'):
    """Plot the target curve in red and the single best matching curve in blue side-by-side for easy comparison."""
    x = np.arange(len(y_target))
    
    plt.figure(figsize=(16, 6))
    # Plot the target curve
    plt.subplot(1, 2, 1)
    plt.plot(x, y_target, color='red', linewidth=2, label='Target Curve')
    plt.title('Target Curve')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid()
    plt.legend()

    # Plot the best matching curve
    plt.subplot(1, 2, 2)
    plt.plot(x, best_match, color='blue', linewidth=1.5, label='Best Match')
    plt.title(f'Best Match (Metric: {metric})')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_distributions_best_matches(y_target, y_best_matches, metric='mae'):
    """Plot the KDE distributions for the target and best 5 matching curves on a single plot."""
    curves = y_best_matches[::-1] + [y_target]
    pal = sns.color_palette('coolwarm', len(curves))
    
    plt.figure(figsize=(15, 8))
    for i, (curve, color) in enumerate(zip(curves, pal)):
        sns.kdeplot(curve, fill=True, alpha=0.5, linewidth=1.5, color=color, label=f'Target' if i == len(curves) - 1 else f'Match {len(curves) - 1 - i}')
    
    plt.xlabel('Time', fontsize=15, fontweight='bold')
    plt.ylabel('Density', fontsize=15, fontweight='bold')
    plt.title(f'Distributions of Target and Best Matches (Metric: {metric})', fontsize=20, fontweight='bold')
    plt.legend()
    plt.grid()
    plt.show()


# plot_utils.py


def plot_time_series(x_data, y_data, title="Original Time Series"):
    """
    Plot a generic time series with x_data and y_data.
    Returns a matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_data, y_data, color='blue', label='Time Series')
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_cleaned_series(x_data, y_data, cleaned_y, noise):
    """
    Plot the original time series, cleaned series, and fill-between for the noise.
    Returns a matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_data, y_data, color='blue', alpha=0.5, label='Original')
    ax.plot(x_data, cleaned_y, color='red', label='Cleaned')
    ax.fill_between(x_data, cleaned_y, y_data, color='gray', alpha=0.3, label='Noise')
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Cleaned Time Series and Noise")
    ax.legend()
    return fig


def plot_decomposition_result(result):
    """
    The seasonal_decompose result object can directly plot
    a figure showing observed, trend, seasonal, and residual.
    Returns that figure.
    """
    fig = result.plot()
    return fig
