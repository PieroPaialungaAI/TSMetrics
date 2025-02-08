from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score
from dtw import accelerated_dtw
from scipy.stats import pearsonr, spearmanr, kendalltau, wasserstein_distance
from scipy.spatial.distance import euclidean, jensenshannon
import numpy as np

def mae(y_true, y_pred):
    """Mean Absolute Error (MAE)"""
    return mean_absolute_error(y_true, y_pred)

def mse(y_true, y_pred):
    """Mean Squared Error (MSE)"""
    return mean_squared_error(y_true, y_pred)

def rmse(y_true, y_pred):
    """Root Mean Squared Error (RMSE)"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def msle(y_true, y_pred):
    """Mean Squared Logarithmic Error (MSLE)"""
    return mean_squared_log_error(y_true, y_pred)

def rmsle(y_true, y_pred):
    """Root Mean Squared Logarithmic Error (RMSLE)"""
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def medae(y_true, y_pred):
    """Median Absolute Error (MedAE)"""
    return median_absolute_error(y_true, y_pred)

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (MAPE)"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error (SMAPE)"""
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def r2(y_true, y_pred):
    """Coefficient of Determination (R² Score)"""
    return r2_score(y_true, y_pred)

def rae(y_true, y_pred):
    """Relative Absolute Error (RAE)"""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true)))

def rse(y_true, y_pred):
    """Relative Squared Error (RSE)"""
    return np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

def mbd(y_true, y_pred):
    """Mean Bias Deviation (MBD)"""
    return np.mean(y_pred - y_true)

def dtw_distance(y_true, y_pred):
    """Dynamic Time Warping (DTW) Distance"""
    dtw_dist, _, _, _ = accelerated_dtw(y_true, y_pred, dist='euclidean')
    return dtw_dist

def pearson_corr(y_true, y_pred):
    """Pearson Correlation Coefficient"""
    return pearsonr(y_true, y_pred)[0]

def spearman_corr(y_true, y_pred):
    """Spearman Rank Correlation"""
    return spearmanr(y_true, y_pred)[0]

def kendall_tau(y_true, y_pred):
    """Kendall’s Tau Correlation"""
    return kendalltau(y_true, y_pred)[0]


def jsd(y_true, y_pred):
    """Jensen-Shannon Divergence"""
    return jensenshannon(y_true, y_pred)

def wasserstein(y_true, y_pred):
    """Earth Mover’s Distance (Wasserstein Distance)"""
    return wasserstein_distance(y_true, y_pred)

# Optional: Dictionary to easily access all metrics by name
METRICS = {
    "mae": mae,
    "mse": mse,
    "rmse": rmse,
    "msle": msle,
    "rmsle": rmsle,
    "medae": medae,
    "mape": mape,
    "smape": smape,
    "r2": r2,
    "rae": rae,
    "rse": rse,
    "mbd": mbd,
    "pearson_corr": pearson_corr,
    "spearman_corr": spearman_corr,
}


