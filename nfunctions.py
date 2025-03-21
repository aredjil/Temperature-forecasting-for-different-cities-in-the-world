import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def load_data(file_path):
    """
    Loads the dataset from a CSV file using pandas.
    
    Parameters:
        file_path (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    df = pd.read_csv(file_path)
    return df

def prepare_data(df, mode="city-year", include_location=False):
    """
    Prepares the data based on the specified mode.
    
    For "city-year" mode, it uses each row as a unique city-year record.
    For "city" mode, it aggregates data by city, averaging the numerical features over the years.
    
    Selected features for clustering:
        - AverageTemperature
        - NO2 Mean
        - SO2 Mean
        - CO Mean
        - Avg_CO2_natural_pross
    Optionally, Latitude and Longitude can be added.
    
    Parameters:
        df (pd.DataFrame): The original dataset.
        mode (str): Either "city-year" or "city".
        include_location (bool): Whether to include Latitude and Longitude.
        
    Returns:
        pd.DataFrame: Preprocessed data ready for clustering.
    """
    # Define the base list of features to use
    features = ["AverageTemperature", "NO2 Mean", "SO2 Mean", "CO Mean", "Avg_CO2_natural_pross"]
    
    # Optionally include geographic features
    if include_location:
        features.extend(["Latitude", "Longitude"])
    
    if mode == "city-year":
        # For city-year mode, we assume each row represents a unique city-year pair.
        # We extract the relevant columns along with City and Year for reference.
        df_prepared = df[["City", "Year"] + features].copy()
    elif mode == "city":
        # For city-level aggregation, group by city and average the numerical columns.
        # Note: We ignore the Year column when aggregating.
        df_prepared = df.groupby("City", as_index=False)[features].mean()
    else:
        raise ValueError("Mode must be either 'city-year' or 'city'.")
        
    return df_prepared


def scale_features(df, feature_columns):
    """
    Scales the selected numerical features using StandardScaler.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the features.
        feature_columns (list): List of column names to be scaled.
        
    Returns:
        scaler (StandardScaler): The fitted scaler.
        X_scaled (np.ndarray): The scaled feature array.
    """
    scaler = StandardScaler()
    X = df[feature_columns].values
    X_scaled = scaler.fit_transform(X)
    return scaler, X_scaled

def evaluate_clusters(X_scaled, max_k=10):
    """
    Evaluates K-means clustering for a range of cluster numbers (from 2 to max_k).
    It calculates the Within-Cluster Sum-of-Squares (inertia) for the Elbow Method and
    the average Silhouette Score for each K.
    
    Generates two plots:
      - "elbow_method.png": Inertia vs. number of clusters.
      - "silhouette_scores.png": Average silhouette score vs. number of clusters.
    
    Parameters:
        X_scaled (np.ndarray): The scaled feature array.
        max_k (int): The maximum number of clusters to try.
        
    Returns:
        tuple: Two lists (inertias and silhouette_scores) corresponding to each tested K.
    """
    inertias = []
    silhouette_scores = []
    k_values = range(2, max_k + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
    
    # Plot the Elbow Method: Inertia vs. number of clusters
    plt.figure()
    plt.plot(list(k_values), inertias, marker='o')
    plt.title("Elbow Method For Optimal k")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Within-Cluster Sum of Squares (Inertia)")
    plt.savefig("elbow_method.png")
    plt.close()
    
    # Plot the Silhouette Scores: Average score vs. number of clusters
    plt.figure()
    plt.plot(list(k_values), silhouette_scores, marker='o', color='orange')
    plt.title("Silhouette Scores For Different k")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Average Silhouette Score")
    plt.savefig("silhouette_scores.png")
    plt.close()
    
    return inertias, silhouette_scores

def perform_kmeans(X_scaled, n_clusters):
    """
    Performs K-means clustering on the scaled data.
    
    Parameters:
        X_scaled (np.ndarray): The scaled feature array.
        n_clusters (int): The number of clusters to form.
        
    Returns:
        tuple: The fitted KMeans model and the array of cluster labels.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    return kmeans, labels

def visualize_clusters(X_scaled, labels):
    """
    Visualizes the clustering results by reducing the dimensionality to 2 components using PCA,
    and then creating a scatter plot of the results colored by cluster labels.
    
    The plot is saved as "clusters.png".
    
    Parameters:
        X_scaled (np.ndarray): The scaled feature array.
        labels (np.ndarray): The cluster labels for each sample.
    """
    # Reduce data to 2 dimensions using PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure()
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50)
    plt.title("K-means Clusters Visualized with PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter, label='Cluster')
    plt.savefig("clusters.png")
    plt.close()

def export_results(df_prepared, labels, mode):
    """
    Exports the clustering results to a CSV file.
    
    The results include the original identifying columns (City and Year if available)
    and an additional 'Cluster' column.
    
    Output file:
      - "city_year_clusters.csv" if mode is "city-year"
      - "city_clusters.csv" if mode is "city"
    
    Parameters:
        df_prepared (pd.DataFrame): The prepared DataFrame used for clustering.
        labels (np.ndarray): The cluster labels.
        mode (str): Either "city-year" or "city".
    """
    # Add the cluster labels as a new column
    df_results = df_prepared.copy()
    df_results['Cluster'] = labels
    
    if mode == "city-year":
        output_file = "city_year_clusters.csv"
    elif mode == "city":
        output_file = "city_clusters.csv"
    else:
        raise ValueError("Mode must be either 'city-year' or 'city'.")
    
    df_results.to_csv(output_file, index=False)
    print(f"Clustering results exported to {output_file}")


def run_clustering(mode="city-year", include_location=False, n_clusters=3, max_k=10, input_file="averaged_output.csv"):
    """
    Runs the complete K-means clustering pipeline. This function is designed for interactive use
    in a Jupyter Notebook. You can modify the parameters to suit your analysis needs.
    
    Parameters:
        mode (str): "city-year" for each city-year pair or "city" for city-level aggregation.
        include_location (bool): Whether to include Latitude and Longitude features.
        n_clusters (int): The number of clusters for K-means.
        max_k (int): The maximum number of clusters to test for evaluation.
        input_file (str): Path to the input CSV file.
    """
    # Load the dataset
    df = load_data(input_file)
    
    # Prepare the data based on selected mode and feature options
    df_prepared = prepare_data(df, mode=mode, include_location=include_location)
    
    # Define the feature columns to be used for scaling and clustering.
    feature_columns = ["AverageTemperature", "NO2 Mean", "SO2 Mean", "CO Mean", "Avg_CO2_natural_pross"]
    if include_location:
        feature_columns.extend(["Latitude", "Longitude"])
    
    # Scale the features using StandardScaler
    scaler, X_scaled = scale_features(df_prepared, feature_columns)
    
    # Evaluate clusters using the Elbow Method and Silhouette Score
    print("Evaluating optimal number of clusters...")
    inertias, silhouette_scores = evaluate_clusters(X_scaled, max_k=max_k)
    print("Elbow Method and Silhouette Score plots have been saved as 'elbow_method.png' and 'silhouette_scores.png'.")
    
    # Perform K-means clustering with the specified number of clusters
    print(f"Performing K-means clustering with {n_clusters} clusters...")
    kmeans, labels = perform_kmeans(X_scaled, n_clusters=n_clusters)
    
    # Visualize the clusters using PCA for dimensionality reduction
    visualize_clusters(X_scaled, labels)
    print("Cluster visualization saved as 'clusters.png'.")
    
    # Export the clustering results to a CSV file
    export_results(df_prepared, labels, mode)
    return df_prepared, labels



def visualize_geographic_clusters(df_prepared, labels):
    """
    Visualizes the clusters on a geographic scatter plot using the actual
    Latitude and Longitude from the data.
    
    Parameters:
        df_prepared (pd.DataFrame): The DataFrame containing at least the 
                                    'Latitude' and 'Longitude' columns.
        labels (np.ndarray): The cluster labels for each row.
    """
    # Ensure that Latitude and Longitude columns exist in the DataFrame.
    if not {"Latitude", "Longitude"}.issubset(df_prepared.columns):
        raise ValueError("DataFrame must contain 'Latitude' and 'Longitude' columns for geographic visualization.")
    
    plt.figure()
    plt.scatter(df_prepared['Longitude'], df_prepared['Latitude'], 
                c=labels, cmap='viridis', s=50)
    plt.title("Geographic Clusters Visualization")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.colorbar(label="Cluster")
    plt.savefig("geographic_clusters.png")
    plt.close()
    print("Geographic clusters visualization saved as 'geographic_clusters.png'.")
