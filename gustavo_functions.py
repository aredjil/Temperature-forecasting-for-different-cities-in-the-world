# Gustavo functions
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from sklearn.feature_selection import mutual_info_regression
from dcor import distance_correlation


def hello_gustavo():
    print('Hello 2 Gustavo!')


def correlation(data, col1, col2, method='pearson'):
    """
    Compute the correlation between two columns in a DataFrame.
    Parameters:
    - data: pandas DataFrame
    - col1: str, first column name
    - col2: str, second column name
    - method: str, correlation method ('pearson', 'spearman', 'kendall')
    Returns:
    - float: correlation coefficient
    Usage:
    correlation_temp_lat = correlation(df, 'AverageTemperature', 'Latitude', method='spearman')
    """
    return data[col1].corr(data[col2], method=method)
