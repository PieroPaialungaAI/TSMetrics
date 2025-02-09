from standard_metrics import METRICS
from constants import * 
import numpy as np
import pandas as pd
from plot_utils import plot_comparison, plot_best_match, plot_distributions_best_matches

class TimeSeriesComparison:
    def __init__(self, y_target, y_pred=None, x=None):
        self.y_target = y_target
        self.y_pred = y_pred
        self.m = len(y_target)
        self.standard_metrics_list = list(METRICS.keys())
        self.closest_curves = {m: None for m in METRICS}
        self.x = np.linspace(0, self.m, self.m) if x is None else x
        self.y_target += 10**-6
        if y_pred is not None:
            self.n = len(y_pred)
            self.metrics_summary = {f'Time Series {i}': {} for i in range(self.n)}
            self.y_pred += 10**-6


    def compute_standard_metrics(self, metric_name, y_pred):
        metric_func = METRICS.get(metric_name)
        if metric_func:
            return metric_func(self.y_target, y_pred)
        raise ValueError(f"Metric '{metric_name}' not found.")

    def compute_standard_metrics_summary(self):
        for i in range(self.n):
            for metric in self.standard_metrics_list:
                try:
                    self.metrics_summary[f'Time Series {i}'][metric] = self.compute_standard_metrics(metric, self.y_pred[i])
                except Exception as e:
                    self.metrics_summary[f'Time Series {i}'][metric] = f"Error: {str(e)}"
        self.metrics_df = pd.DataFrame(self.metrics_summary).transpose()
        return self.metrics_df

    def plot_comparison(self, num_to_plot=10, metric='mae'):
        distances = [METRICS[metric](self.y_target, y) for y in self.y_pred]
        sorted_indices = np.argsort(distances)
        closest_curves = [self.y_pred[i] for i in sorted_indices[:num_to_plot]]
        self.closest_curves[metric] = closest_curves
        plot_comparison(self.y_target, self.y_pred, closest_curves, metric)

    def plot_best_match(self, metric='mae'):
        if not self.closest_curves[metric]:
            distances = [METRICS[metric](self.y_target, y) for y in self.y_pred]
            sorted_indices = np.argsort(distances)
            self.closest_curves[metric] = [self.y_pred[i] for i in sorted_indices]
        best_match = self.closest_curves[metric][0]
        plot_best_match(self.y_target, best_match, metric)

    def plot_distributions_best_matches(self, metric='mae'):
        MIN_BEST_RIDGELINE = 5
        if not self.closest_curves[metric]:
            distances = [METRICS[metric](self.y_target, y) for y in self.y_pred]
            sorted_indices = np.argsort(distances)
            self.closest_curves[metric] = [self.y_pred[i] for i in sorted_indices]
        y_best_matches = self.closest_curves[metric][:MIN_BEST_RIDGELINE]
        plot_distributions_best_matches(self.y_target, y_best_matches, metric)
