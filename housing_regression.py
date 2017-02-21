import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans

lmodel = LinearRegression()

# Read in CSV's to pandas
train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
labels = train["SalePrice"]

model_data = pd.concat([train,test],ignore_index=True)
model_data = model_data.drop("SalePrice", 1)

# Some functions

########## KMeans to modify nan values ##########
########## inputs:
########## label of col to update, indices in that col that have nan value,
########## data for clustering (including nan col), k number of clusters

########## output:
########## dictionary of {index : new value to replace nan}
def KMeans_nan_replacement(nan_col_label, nan_indices, k_data, k):
    # Replace nan with mean value for column
    k_data[k_data] = k_data[k_data].fillna(k_data[k_data].mean())

    # first iteration of KMeans
    kmeans = KMeans(n_clusters=k, random_state=0).fit(k_data)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    k_data[nan_indices_lot_frontage]

    prev_labels = []
    converged = False

    # Run KMeans, update previously nan values to converge on their centroids,
    # Until the clusters stop changing
    while(converged == False):
        for nan_index in nan_indices:
            # Set LotFrontage value that was previously nan to
            # centroid for the k_data point at that same index
            k_data[nan_index][0] = centroids[labels[nan_index]][0]

        # Rerun KMeans
        kmeans = KMeans(n_clusters=k, random_state=0).fit(k_data)
        # Store new labels
        labels = kmeans.labels_
        # Check em, boys!
        print labels
        # If labels did not change in this run, algorithm has converged
        # (is this error prone if the window of distance to centroid is large?)
        if np.all(labels == prev_labels):
            converged = True

        prev_labels = labels

    # Return a dict of {index : new value}
    return dict(zip(nan_indices, k_data[[i for i in nan_indices]]))
##############################

# Store id's for rows with nan
nan_indices_lot_frontage = model_data['LotFrontage'].index[model_data['LotFrontage'].apply(np.isnan)]

k = 9
k_data = np.column_stack(model_data.loc[:, "MSSubClass":"OverallCond"].values)

# Drop seemingly useless cols that have nan
model_data = model_data.drop(["GarageYrBlt"], axis=1)