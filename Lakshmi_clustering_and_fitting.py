# Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import sklearn.metrics as skmet
from sklearn.preprocessing import MinMaxScaler
import sklearn.cluster as cluster
from scipy.optimize import curve_fit


# Plotting k clusters using k-means clustering.
def plot_kmeans(scaled_data, n):
    kmeans = cluster.KMeans(n_clusters = n)
    kmeans.fit(scaled_data)
    cen = kmeans.cluster_centers_
    # Printing the centers of the clusters.
    for i in range(n):
        print('The coordinates of the center of cluster', i+1, 'are (', cen[i, 0], ',', cen[i, 1], ')')
    # Printing the silhouette score.
    print('The Silhouette score of the clusters is ', skmet.silhouette_score(scaled_data, kmeans.labels_))

    # Plotting the scaled clusters.
    plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=kmeans.labels_)
    plt.xlabel('Scaled values of the amount of CO2 emitted')
    plt.ylabel('Proportion of people having access to electricity')
    plt.title('K-means clustering')
    for i in range(n):
        plt.plot(cen[i, 0], cen[i, 1], "*", markersize=10, c='r')
    plt.show()
    return kmeans

# Finding a sample from ith cluster.
def find_sample_from_cluster(clusters, n):
    for i in range(len(clusters)):
        if (clusters[i] == n):
            return i

# This function determines the relation ship between x and y values of the courve y=f(x).
def objective(x, a, b):
    return a*x + b

# Curve fitting of CO2 data.
def fit_curve(X, Y):
    popt, _ = curve_fit(objective, X, Y)
    a, b = popt
    plt.scatter(X, Y)
    plt.xlabel('Year')
    plt.ylabel('Percentage of population having accing to electricity')
    plt.title('Curve fitting of percentage of population \n having accing to electricity in Bangladesh')
    x_line = np.arange(min(X), max(X), 1)
    y_line = objective(x_line, a, b)
    plt.plot(x_line, y_line, '--', color='b')
    plt.show()
    # Predicting the future CO2 consumption from the curve.
    future_years = [2024, 2037, 2048, 2049, 2050]
    for i in range(len(future_years)):
        print('The predicted percentage of people in Bangladesh having access to electricity in', future_years[i], 'is', objective(future_years[i], a, b))

# This function plots the elbow plot of different  number of clusters.
def plot_elbow_graph(data):
    data = list(zip(data[:, 0], data[:, 1]))
    inertias = []
    for i in range(10):
        kmeans = cluster.KMeans(n_clusters=i+1)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    plt.plot(range(1,11), inertias, marker='*')
    plt.title('Elbow Graph to determine the optimal number of clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

# Min Max Scaler function changes the range of the data to [0, 1]
def min_max_scaler(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler.transform(data)

# Loading the data
co2_data = pd.read_csv('co2_emission_data.csv')
access_to_electricity_data = pd.read_csv('access_to_electricity_data.csv')
data = pd.DataFrame({'Country Name' : co2_data.iloc[:, 0].values,
                    'CO2 Emission' : co2_data.iloc[:, 63],
                    'Access to electricity' : access_to_electricity_data.iloc[:, 63]})

# Dropping rows with missing values
data = data.dropna()

# Scaling the data so as to eliminate the effect of units
data_scaled = min_max_scaler(data.iloc[:, 1:])

# Drawing the elbow graph to find the optimal number of clusters in K-means algorithm and then plotting the k means clusters.
plot_elbow_graph(data_scaled)
kmeans = plot_kmeans(data_scaled, 3)

# Comparing the two countires of differnt clusters.
first_country = find_sample_from_cluster(kmeans.labels_, 0)
second_country = find_sample_from_cluster(kmeans.labels_, 1)
third_country = find_sample_from_cluster(kmeans.labels_, 2)
print(data.iloc[[first_country, second_country, third_country], :])

# Curve fitting
fit_curve(range(1991, 2021), access_to_electricity_data.iloc[20, 35:65])