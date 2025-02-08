from standard_metrics import *
import pandas as pd
import matplotlib.pyplot as plt 

class TSMetrics:
    def __init__(self, y_target, y_pred = None):
        self.y_target = y_target
        self.y_pred = y_pred
        self.standard_metrics_list = list(METRICS.keys())
        if y_pred:
            self.n = len(self.y_pred)
            self.metrics_summary = { 'Time Series ' + str(i): {} for i in range(self.n) }
            self.y_pred += 10**-6
        self.y_target += 10**-6
        self.closest_curves = {m: None for m in METRICS}
        self.length_curve = len(self.y_target)

    def compute_standard_metrics(self, metric_name, y_pred):
        """Computes the given metric dynamically."""
        metric_func = METRICS.get(metric_name)
        if metric_func:
            return metric_func(self.y_target, y_pred)
        else:
            raise ValueError(f"Metric '{metric_name}' not found. Available metrics: {list(METRICS.keys())}")

    def compute_standard_metrics_summary(self):
        """Compute standard metrics for each time series and store them in a summary dictionary."""
        k = -1
        for i in range(self.n):
            k += 1
            for metric in self.standard_metrics_list:
                try:
                    self.metrics_summary[f'Time Series {i}'][metric] = self.compute_standard_metrics(metric, self.y_pred[k])
                except Exception as e:
                    self.metrics_summary[f'Time Series {i}'][metric] = f"Error: {str(e)}"

        # Convert the summary dictionary to a DataFrame for easy analysis
        self.metrics_df = pd.DataFrame(self.metrics_summary).transpose()
        return self.metrics_df



    def plot_comparison(self, num_to_plot = 10, metric='mae'):
        """Plot the target curve in red and the closest comparison curves in grey based on a given metric.

        Args:
            y_target (array-like): The target time series.
            y_pred (list of array-like): List of predicted time series.
            metric (function): The metric function to compute distances (default is Mean Absolute Error).
        """
        # Compute distances using the specified metric
        distances = [METRICS[metric](self.y_target, y) for y in self.y_pred]
        
        # Sort by distance and select the closest curves
        sorted_indices = np.argsort(distances)
        closest_curves = [self.y_pred[i] for i in sorted_indices[:num_to_plot]]
        self.closest_curves[metric] = closest_curves
        # Plot target curve in red and closest comparison curves in grey
        plt.figure(figsize=(12, 6))
        x = np.arange(len(self.y_target))
        for curve in closest_curves:
            plt.plot(x, curve, color='grey', alpha=1., linewidth=0.8)
        plt.plot(x, self.y_target, color='red', linewidth=2, label='Target Curve')
        plt.title(f'Comparison of Target Curve and Closest {num_to_plot} Curves (Metric: {metric})')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.grid()
        plt.legend()
        plt.show()



    def plot_best_match(self, metric='mae'):
        """Plot the target curve in red and the single best matching curve in blue side-by-side for easy comparison.

        Args:
            metric (str): The metric name to compute distances (default: 'mae').
        """
        if metric not in METRICS:
            raise ValueError(f"Metric '{metric}' not found. Available metrics: {list(METRICS.keys())}")

        # Compute distances using the specified metric
        if not self.closest_curves[metric]:
            distances = [METRICS[metric](self.y_target, y) for y in self.y_pred]
            sorted_distances = np.argsort(distances)
            self.closest_curves[metric] = self.y_pred[sorted_distances]        
            # Find the best match (smallest distance)
        best_match = self.closest_curves[metric][0]

        # Plot target and best matching curve side-by-side
        x = np.arange(len(self.y_target))
        
        plt.figure(figsize=(16, 6))
        
        # Plot the target curve
        plt.subplot(1, 2, 1)
        plt.plot(x, self.y_target, color='red', linewidth=2, label='Target Curve')
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

    


