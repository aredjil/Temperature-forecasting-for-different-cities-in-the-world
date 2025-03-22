# Report:

Here we describe the work that we did about the project number 3 **Temperature-forecasting-for-different-cities-in-the-world**

## Members team:
We clean the data all togueter.

Nicola  nicola@ictp.it : clustering 

Bello bello@ictp.it : Predicting 

Gustavo Paredes gparede@ictp.it : correlation 


## Temperature forecasting for different cities in the world

## Introduction
For this project you are asked to analyze three datasets, called respectively:
1. pollution_us_2000_2016.csv
2. greenhouse_gas_inventory_data_data.csv
3. GlobalLandTemperaturesByCity.csv

You are asked to extract from dataset 2 only the US countries (for which we have info in the other datasets) and to perform the following tasks:
- to measure how pollution and temperature create cluster tracing the high populated cities in the world
- to analyze the correlation between pollution data and temperature change.
- to predict the yearly temperature change of a given city over a given time period, using the <b>ARIMA model</b> for <b>time series forecasting</b>, that is a model for time series forecasting integrating AR models with Moving Average.
- (OPTIONAL) rank the 5 cities that will have a highest temperature change in US


### TASK1 :Cluster Analysis
You use K-means or DBSCAN to perform the cluster analysis, and create a new dataset where the cities are associated to the different identified clusters

### TASK 2: Correlation Analysis

You measure the correlation between:
- temperature and latitude
- temperature and pollution
- temperature change (difference between the average temperature measured over the last 3 years and the previous temperature) and pollution

~~~
~~~
#### Discution

We need to measure correlations between different variables in the dataset (df). Correlation tells us how strongly two variables are related (positive, negative, or no correlation).
1 Correlation between Temperature and Latitude

- The idea is to check whether temperature varies with latitude (higher latitudes usually mean lower temperatures).

For this part we compare the correlation of Pearson, Spearman and Kendall and we find that temperature and length are indeed correlated as shown in the following graph.

<div align="center">
    <img src="figures/temperature_vs_latitude.png" alt="Temperature vs Latitude" width="500">
</div>


- We'll use Pearson correlation (df.corr()) to measure the linear relationship between AverageTemperature and Latitude.

- A negative correlation means that temperature decreases as latitude increases (which is expected since higher latitudes are colder).

2 Correlation between Temperature and Pollution

- We want to check how pollution affects temperature.

- We'll analyze the correlation between AverageTemperature and different pollution indicators (NO2 Mean, SO2 Mean, CO Mean).

- A positive correlation would mean that pollution is linked to higher temperatures.

3 Correlation between Temperature Change and Pollution

- We need to calculate temperature change over the last three years for each city.

	- Example: For Year = 2013, we compute the difference between the average temperature of the last 3 years (2011-2013) and the previous period (2008-2010).

- Then, we check the correlation between temperature change and pollution levels to see if pollution is affecting long-term temperature trends.
~~~

~~~
### TASK 3: Predicting the Temperature of a Given City across a Specified Time Period
After reading the data in the temperature data set, for each city cluster, before applying the ARIMA model you perform the following steps:

- EDA
- data cleaning and preprocessing (Converting the 'dt' (date) column to DateTime format, removing NaN)
- feature selection
- make the time-series stationary
- check for stationarity : Calculating the Augmented Dickey-Fuller Test statistic 
- identify the (p, q) order of the ARIMA model using ACF partial autocorrelation plot

Then:

-fit the ARIMA model using the calculated p, q values.
-calculate the MSE with respect to the true temp. measurements to estimate the performance of the model


NOTE: ARIMA models need the data to be stationary i.e. the data must not exhibit trend and/or seasonality. To identify and remove trend and seasonality, we can use
- seasonal decomposition
- differencing

