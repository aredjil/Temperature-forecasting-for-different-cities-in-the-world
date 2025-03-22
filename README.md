# Temperature Forecasting for Different Cities Worldwide

## Authors
- Nicola
- Bello
- Gustavo Paredes Torres

## Project Overview
This project aims to analyze temperature trends in different cities worldwide, perform clustering analysis, evaluate correlations between temperature and pollution, and forecast future temperature variations using time series analysis.

## Objectives
- **Data Cleaning:** Preprocessing raw temperature and pollution data.
- **Clustering Analysis:** Applying K-means or DBSCAN to identify city clusters based on temperature patterns.
- **Correlation Analysis:** Measuring the relationship between temperature and pollution.
- **Time Series Forecasting:** Predicting temperature changes using ARIMA models.
- **Visualization:** Generating informative figures to support the analysis.

## Report

The detailed report of this project can be found [here](README.md).

## Project Structure

```
.
├── Project_3
│   └── data
├── README.md
├── Report.md   
│   
├── codes
│   ├── Cleaning_data_Task1.ipynb
│   ├── Correlation_analysis.ipynb
│   ├── Project3_nicola.ipynb
│   ├── Tests.ipynb
│   ├── g_functions.py
│   └── nfunctions.py
├── data-project3
│   ├── Clean_data
│   │   ├── averaged_output.csv
│   │   ├── city_year_clusters.csv
│   │   ├── clean_data.csv
│   │   ├── cleaned_temperature_pollution_data.csv
│   │   └── new_green.csv
│   └── Original_data
│       ├── GlobalLandTemperaturesByCity.csv.zip
│       ├── greenhouse_gas_inventory_data_data(1).csv
│       └── pollution_us_2000_2016.csv.zip
├── figures
    ├── clusters.png
    ├── correlation_heatmap.png
    ├── correlation_pollution_methods.png
    ├── correlation_pollution_methods_per_3year.png
    ├── elbow_method.png
    ├── geographic_clusters.png
    ├── pollution_vs_temperature.png
    ├── silhouette_scores.png
    └── temperature_vs_latitude.png
```