import pandas as pd
import numpy as np
from sklearn.cluster import KMeans



def load_data(filename):
    data = pd.read_csv(filename)
    return data

def Kmeans(data, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    return kmeans.labels_