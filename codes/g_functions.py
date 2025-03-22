# Gustavo functions
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from sklearn.feature_selection import mutual_info_regression
from dcor import distance_correlation
import pandas as pd
import seaborn as sns


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


# Function to compute multiple correlations for a given target variable
def compute_all_correlations(data, target_col, variables):
    correlation_results = {}
    for var in variables:
        correlation_results[var] = {
            "Pearson": correlation(data, target_col, var, method="pearson"),
            "Spearman": correlation(data, target_col, var, method="spearman"),
            "Kendall": correlation(data, target_col, var, method="kendall"),
            "Distance Correlation": distance_correlation(data[target_col].dropna(), data[var].dropna())
        }
    return pd.DataFrame(correlation_results).T

# Function to plot correlation results


def plot_correlation_results(correlation_df, title):
    correlation_df.plot(kind='bar', figsize=(
        10, 6), colormap="coolwarm", edgecolor="black")
    plt.xlabel("Variable")
    plt.ylabel("Correlation Score")
    plt.title(title)
    plt.axhline(0, color='black', linewidth=1)
    plt.ylim(-1, 1)
    plt.xticks(rotation=0)
    plt.legend(title="Method")
    plt.show()

# Function to plot scatter plots for temperature vs. pollution variables


def plot_scatterplots(data, variables, target_col):
    fig, axes = plt.subplots(
        1, len(variables), figsize=(6 * len(variables), 5))
    for i, var in enumerate(variables):
        sns.scatterplot(ax=axes[i], x=data[var], y=data[target_col], alpha=0.5)
        axes[i].set_xlabel(var)
        axes[i].set_ylabel(target_col)
        axes[i].set_title(f"{target_col} vs. {var}")
    plt.tight_layout()
    plt.show()
