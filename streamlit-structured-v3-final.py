#!/usr/bin/env python
# coding: utf-8
#get_ipython().system('pip install streamlit')

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Read the data
df_inflation = pd.read_csv('../../../Documents/GitHub/BI-Fall-2023-Exam-Project/Data/US_inflation_rates.csv')
df_inflation['date'] = pd.to_datetime(df_inflation['date'])
df_inflation['year'] = df_inflation['date'].dt.year
year_df_inflation = df_inflation[(df_inflation['year'] >= 1947) & (df_inflation['year'] <= 2023)]
index_year_df_inflation = year_df_inflation.index
X_df_inflation = np.array(index_year_df_inflation).reshape(-1, 1)
y_df_inflation = df_inflation['value'].values.reshape(-1, 1)

# Create Streamlit web app
st.title('US Inflation Data Analysis')

# Display the DataFrame
#st.subheader('US Inflation Data')
#st.write(df_inflation)

# Create a histogram of the years
sns.distplot(df_inflation['value'], label='value', norm_hist=True)

# ## Train the Inflation data model
# #### Split the Data in Dependent y and Independent X Data Sets
# Splitting the dataset into the Training set and Test set
X_train_df_inflation, X_test_df_inflation, y_train_df_inflation, y_test_df_inflation = train_test_split(X_df_inflation, y_df_inflation, test_size=0.2, random_state=0)

# Fitting Linear Regression to the dataset
lin_reg_df_inflation = LinearRegression()
lin_reg_df_inflation.fit(X_df_inflation, y_df_inflation)
# Visualizing the Linear Regression results
def viz_linear_inflation():
    fig, ax = plt.subplots()
    ax.scatter(X_df_inflation, y_df_inflation, color='red')
    ax.plot(X_df_inflation, lin_reg_df_inflation.predict(X_df_inflation), color='blue')
    ax.set_title('Linear Regression')
    ax.set_xlabel('year in index form (1947-2023)')
    ax.set_ylabel('value')
    return fig

# Visualize the Linear Regression
st.subheader('Linear Regression')
st.pyplot(viz_linear_inflation())

############################# Visualize inflation polynomial regression #####################################

# Fitting Polynomial Regression to the dataset
poly_model_df_inflation = PolynomialFeatures(degree=5)
X_poly_df_inflation = poly_model_df_inflation.fit_transform(X_df_inflation)
pol_reg_df_inflation = LinearRegression()
pol_reg_df_inflation.fit(X_poly_df_inflation, y_df_inflation)
y_predict_df_inflation = pol_reg_df_inflation.predict(X_poly_df_inflation)
# Define the viz_polymonial_inflation function
def viz_polymonial_inflation():
    fig, ax = plt.subplots()
    ax.scatter(X_df_inflation, y_df_inflation, color='red')
    ax.plot(X_df_inflation, y_predict_df_inflation , color='blue')
    ax.set_title('Polynomial Regression')
    ax.set_xlabel('year in index form (1947-2023)')
    ax.set_ylabel('value')
    return fig

# Visualize the Polynomial Regression
st.subheader('Polynomial Regression')
st.pyplot(viz_polymonial_inflation())

########################### R2_score polynomial regression ##################################
# Calculate R-squared score for Polynomial Regression
r2_score_poly_df_inflation = r2_score(y_df_inflation, y_predict_df_inflation)
# Calculate and display the R-squared score for Polynomial Regression
st.subheader('R-squared (R^2) Score for Polynomial Regression')
st.write(f'R-squared (R^2) score for Polynomial Regression: {r2_score_poly_df_inflation:.5f}')

#################################################################################

# Create a mapping of index values to years
years_map = {}
for i in range(0, len(X_df_inflation)):
    year = 1947 + i // 12  # Assuming each index represents a month
    years_map[i] = year

# Predicting a new result with Linear Regression
st.subheader('Predict with Linear Regression')
x_linear = st.slider('Select a year for prediction (Linear Regression)', min(years_map.keys()), max(years_map.keys()), 200)
linear_result = lin_reg_df_inflation.predict([[x_linear]])
selected_year = years_map[x_linear]
st.write(f'Predicted value for the year {selected_year}: {linear_result[0][0]}')




# Create a mapping of index values to years
years_map = {}
for i in range(0, 1009):
    year = 1947 + i // 12  # Assuming each index represents a month
    years_map[i] = year

# Predicting a new result with Polynomial Regression
st.subheader('Predict with Polynomial Regression')
x_poly = st.slider('Select a year for prediction (Polynomial Regression)', min(years_map.keys()), max(years_map.keys()), 996)
yearly_prediction_in_index_df_inflation = pol_reg_df_inflation.predict(poly_model_df_inflation.transform([[x_poly]]))
selected_year = years_map[x_poly]
st.write(f'Predicted value for the year {selected_year}: {yearly_prediction_in_index_df_inflation[0][0]}')


######################## Predict ##################################

# Create a DataFrame for years 2023.5 to 2030
years_2023_to_2030_df_inflation = np.arange(2023.5, 2032.5).reshape(-1, 1)
X_pred_df_inflation = poly_model_df_inflation.transform(years_2023_to_2030_df_inflation)
y_pred_df_inflation = pol_reg_df_inflation.predict(X_pred_df_inflation)

# Calculate the percentage growth
percentage_growth_df_inflation = (y_pred_df_inflation - y_pred_df_inflation[0]) / y_pred_df_inflation[0] * 10

# Define a function to create the plot
def create_percentage_growth_plot():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(2023.5, 2032.5), percentage_growth_df_inflation, marker='o', linestyle='-', color='b')
    ax.set_title('Predicted Percentage Growth of US Inflation Rate (2023-2030)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Percentage Growth')
    ax.set_ylim(0, .3)  # Set the y-axis limits from 0 to 0.3
    ax.grid(True)
    return fig

# Visualize the Percentage Growth Plot
st.subheader('Percentage Growth Plot')
st.pyplot(create_percentage_growth_plot())

