# -*- coding: utf-8 -*-
"""
Created on Wed May 8 20:12:50 2023

@author: sneha
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
from scipy.optimize import curve_fit


def preprocess_GDP_data():
    """
    Preprocesses GDP data for clustering and forecasting.
    Reads GDP per capita and GDP current GDP data from CSV files, drops rows \
        with missing data for 2021, 
    selects relevant columns, merges the GDP per capita and GDP current data on \
        country name, renames columns, 
    normalizes data using StandardScaler, performs k-means clustering,\
        and adds cluster labels to the DataFrame.

    Returns
    -------
    df_2021 : pandas DataFrame
        A merged DataFrame of GDP per capita and GDP current data for 2021,\
            along with a column of cluster labels.
    """
    df_gdp_pc = pd.read_csv("GDP per capita (current US$).csv", skiprows=4)
    df_gdp_current = pd.read_csv("GDP (current US$).csv", skiprows=4)

    # drop rows with nan's in 2021
    df_gdp_pc = df_gdp_pc[df_gdp_pc["2021"].notna()]
    df_gdp_current = df_gdp_current.dropna(subset=["2021"])

    # select relevant columns
    df_gdp_pc2021 = df_gdp_pc[["Country Name", "Country Code", "2021"]].copy()
    df_gdp_current2021 = df_gdp_current[["Country Name", "Country Code", "2021"]].copy()

    # merge GDP per capita and GDP current data on country name
    df_gdp2021 = pd.merge(df_gdp_pc2021, df_gdp_current2021, on="Country Name", \
                       how="outer")

    # drop rows with missing data
    df_gdp2021 = df_gdp2021.dropna()

    # rename columns
    df_gdp2021 = df_gdp2021.rename(columns={"2021_x":"GDP per capita", "2021_y":"GDP current"})

    # normalize data using StandardScaler
    scaler = StandardScaler()
    #df_cluster = scaler.fit_transform(df_gdp2021)
    df_cluster = scaler.fit_transform(df_gdp2021[["GDP per capita", "GDP current"]])

    # perform k-means clustering and add cluster labels to DataFrame
    kmeans = cluster.KMeans(n_clusters=5)
    kmeans.fit(df_cluster)
    df_gdp2021["labels"] = kmeans.labels_
    
    # Apply elbow curve method to determine optimal number of clusters
    distortions = []
    K = range(1, 11)
    for k in K:
        kmeans = cluster.KMeans(n_clusters=k)
        kmeans.fit(df_cluster)
        distortions.append(kmeans.inertia_)
    
    # Plot the elbow curve
    plt.figure(figsize=(8, 6))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Elbow Curve')
    plt.show()

    return df_gdp2021


def forecast_GDP():
    """
    Reads GDP data, fits a logistic function to the data,\
        and plots the forecast.
    """
    
    # read data from CSV file
    df_gdp = pd.read_csv("GDP growth (annual %).csv", skiprows=4)
    df_gdp.head()
    df_gdp = df_gdp.set_index('Country Name', drop=True)
    df_gdp = df_gdp.loc[:,'1960':'2021']
    df_gdp = df_gdp.transpose()
    df_gdp = df_gdp.loc[:,'India']
    df = df_gdp.dropna(axis=0)

    # create DataFrame with year and GDP columns
    df_gdp=pd.DataFrame()
    df_gdp['Year']=pd.DataFrame(df.index)
    df_gdp['GDP']=pd.DataFrame(df.values)

    # fit logistic function to data
    def logistic(t, n0, g, t0):
        """Calculates the logistic function with scale factor \
            n0 and growth rate g"""
        f = n0 / (1 + np.exp(-g*(t - t0)))
        return f

    df_gdp["Year"] = pd.to_numeric(df_gdp["Year"])
    param, covar = curve_fit(logistic, df_gdp["Year"], \
                             df_gdp["GDP"], \
                                 p0=(1.2e12, 0.03, 1990.0), maxfev=50000)

    # generate forecast using logistic function
    year = np.arange(1960, 2031)
    forecast = logistic(year, *param)

    # generate predictions for 3 years
    year_pred = np.arange(2022, 2025)
    forecast_pred = logistic(year_pred, *param)

    # calculate confidence interval
    stderr = np.sqrt(np.diag(covar))
    conf_interval = 1.96 * stderr
    upper = logistic(year, *(param + conf_interval))
    lower = logistic(year, *(param - conf_interval))

    # plot the data, forecast, and predictions
    plt.figure()
    plt.plot(df_gdp["Year"], df_gdp["GDP"], label="GDP")
    plt.plot(year, forecast, label="Forecast")
    plt.plot(year_pred, forecast_pred, 'ro', label="Predictions")
    plt.fill_between(year, upper, lower, alpha=0.2, label="Confidence Interval")
    plt.xlabel("Year")
    plt.ylabel("GDP")
    plt.title("Forecast")
    plt.legend()
    plt.show()


df_2021 = preprocess_GDP_data()

# perform k-means clustering to obtain cluster centers
kmeans = cluster.KMeans(n_clusters=5)
kmeans.fit(df_2021[["GDP per capita", "GDP current"]])
centers = kmeans.cluster_centers_

# plot all clusters and centers in one plot
plt.figure(figsize=(6, 5))
cm = plt.cm.get_cmap('tab10')
for i, label in enumerate(np.unique(df_2021["labels"])):
    plt.scatter(df_2021[df_2021["labels"] == label]["GDP per capita"], \
                df_2021[df_2021["labels"] == label]["GDP current"], \
                    10, label="Cluster {}".format(label), cmap=cm, alpha=0.7)
plt.scatter(centers[:,0], centers[:,1], 50, "k", marker="D", \
            label="Cluster centers")
plt.xlabel("GDP per capita")
plt.ylabel("GDP current")
plt.title("Kmeans Clustering")
plt.legend()
plt.show()

forecast_GDP()