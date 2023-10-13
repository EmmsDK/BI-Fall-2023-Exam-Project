import streamlit as st
import plotly.graph_objects as go

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Load your dataset (modify the path accordingly)
df_salary_ini = pd.read_csv('../../../Documents/GitHub/BI-Fall-2023-Exam-Project/Data/ds_salaries2.csv')

# Function to clean and extract data
def clean_and_extract_data(df):
    # Initialize empty arrays for storing the matching rows
    work_year_values = []
    job_title_values = []
    employee_residence_values = []
    employment_type_values = []
    salary_in_usd_values = []
    experience_level_values = []
    remote_ratio_values = []

    # Criteria for filtering the data
    for index, row in df.iterrows():
        employee_residence = row['employee_residence']
        employment_type = row['employment_type']
        job_title = row['job_title']
        experience_level = row['experience_level']

        if (
            employee_residence == 'US' and
            employment_type == 'FT' and
            job_title in ['Data Scientist', 'Data Engineer', 'Data Analyst', 'Machine Learning Engineer'] and
            experience_level in ['EN', 'MI', 'SE', 'EX']
        ):
            # Save the values from the matching rows into arrays
            work_year_values.append(row['work_year'])
            job_title_values.append(job_title)
            employee_residence_values.append(employee_residence)
            employment_type_values.append(employment_type)
            salary_in_usd_values.append(row['salary_in_usd'])
            experience_level_values.append(experience_level)
            remote_ratio_values.append(row['remote_ratio'])

    # Create a DataFrame from the matching data
    data = {
        'work_year': work_year_values,
        'job_title': job_title_values,
        'employee_residence': employee_residence_values,
        'employment_type': employment_type_values,
        'salary_in_usd': salary_in_usd_values,
        'experience_level': experience_level_values,
        'remote_ratio': remote_ratio_values
    }
    matching_df = pd.DataFrame(data)

    return matching_df

st.title("Data Analyst Job Count")

df_sal = clean_and_extract_data(df_salary_ini)

data_analyst_count = df_sal['job_title'].value_counts().get('Data Analyst', 0)
st.write(f"Number of Data Analyst jobs: {data_analyst_count}")

data_engineer_count = df_sal['job_title'].value_counts()['Data Engineer']
st.write(f"Number of Data Engineer jobs: {data_engineer_count}")

data_scientist_count = df_sal['job_title'].value_counts()['Data Scientist']
st.write(f"Number of Data Scientist jobs: {data_scientist_count}")

machine_learning_engineer_count = df_sal['job_title'].value_counts()['Machine Learning Engineer']
st.write(f"Number of Machine Learning Engineer jobs: {machine_learning_engineer_count}")

# Define the functions for training regression models and predicting salaries


st.title("Work Arrangement and Salary Analysis")

# Extract the values for each category
remote = df_sal[df_sal['remote_ratio'] == 100]['remote_ratio'].count()
hybrid = df_sal[df_sal['remote_ratio'] == 50]['remote_ratio'].count()
office = df_sal[df_sal['remote_ratio'] == 0]['remote_ratio'].count()

categories = ['Remote', 'Hybrid', 'Fully In Office']
values = [remote, hybrid, office]

# Create an interactive bar chart using Plotly Express
st.subheader("Distribution of Work Arrangement")
bar_fig = px.bar(x=categories, y=values, labels={'x': 'Work Arrangement', 'y': 'Count'})
st.plotly_chart(bar_fig)

# Create an interactive scatter plot with regression line using Plotly Express
st.subheader("Salary vs. Remote Ratio")
scatter_fig = px.scatter(df_sal, x='remote_ratio', y='salary_in_usd', trendline='ols', labels={'remote_ratio': 'Remote Ratio', 'salary_in_usd': 'Salary in USD'})
st.plotly_chart(scatter_fig)

# Group the data by 'remote_ratio' and calculate the mean salary for each group
average_salaries = df_sal.groupby('remote_ratio')['salary_in_usd'].mean()

# Reset the index to make 'remote_ratio' a column
average_salaries = average_salaries.reset_index()

# Rename the 'salary_in_usd' column to 'average_salary'
average_salaries = average_salaries.rename(columns={'salary_in_usd': 'average_salary'})

# Replace 'remote_ratio' values with descriptive labels
average_salaries['remote_ratio'] = average_salaries['remote_ratio'].replace({0: 'Fully In Office', 50: 'Hybrid', 100: 'Remote'})

# Display the average salaries as a table
st.subheader("Average Salaries by Work Arrangement")
st.dataframe(average_salaries)


# Load your data
df_remoteWork = pd.read_excel('../../../Documents/GitHub/BI-Fall-2023-Exam-Project/Data/statistic_id1356325_us-workers-working-hybrid-or-remote-vs-on-site-2019-q4-2022.xlsx', sheet_name='Data')

# Create a Streamlit app
st.title("Work Arrangement Over Time")

# Create an interactive line chart using Plotly Express
line_fig = px.line(df_remoteWork, x='Period', y=['Hybrid', 'Remote', 'On-site'], labels={'Period': 'Period', 'value': 'Count'})
line_fig.update_traces(mode='markers+lines')  # Add markers to the lines
line_fig.update_layout(title="Hybrid vs. Remote vs. On-site", xaxis_title="Period", yaxis_title="Count")
st.plotly_chart(line_fig)



def train_regression_models(df, job_titles, experience_levels):
    models = {}  # To store trained models

    for job_title_to_match in job_titles:
        models[job_title_to_match] = {}  # Create a sub-dictionary for each job title

        for experience_level in experience_levels:
            if job_title_to_match == "Machine Learning Engineer" and experience_level == "EX":
                # Skip this specific combination
                continue

            work_year_values = []
            salary_in_usd_values = []

            for index, row in df.iterrows():
                job_title = row['job_title']
                employment_type = row['employment_type']
                employee_residence = row['employee_residence']
                current_experience_level = row['experience_level']

                # Checking for matching values
                if (
                    job_title == job_title_to_match
                    and employment_type == 'FT'
                    and employee_residence == 'US'
                    and current_experience_level == experience_level
                ):
                    work_year_values.append(row['work_year'])
                    salary_in_usd_values.append(row['salary_in_usd'])

            # Check if there are enough data points to perform train-test split
            if len(work_year_values) < 2:
                print(f"Insufficient data for {job_title_to_match} and {experience_level}. Skipping.")
                continue

            x = pd.DataFrame(work_year_values, columns=['work_year'])
            y = pd.Series(salary_in_usd_values)

            # Split the data into training and testing sets
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(x_train, y_train)

            # Store the model in the sub-dictionary with a key based on experience level
            models[job_title_to_match][experience_level] = model

    return models

def predict_salaries_for_all_combinations(trained_models, job_titles, experience_levels, work_year_to_predict):
    predictions = {}  # To store predictions

    for job_title in job_titles:
        predictions[job_title] = {}  # Create a sub-d


# Create a Streamlit app and load your data
st.title("Regression Models and Predicted Salaries")

# Define the job titles and experience levels
job_titles = ['Data Scientist', 'Data Engineer', 'Data Analyst', 'Machine Learning Engineer']
experience_levels = ['EN', 'MI', 'SE', 'EX']

# Train the regression models
trained_models = train_regression_models(df_salary_ini, job_titles, experience_levels)

# Plot regression results for your dataset
def plot_regression_results(models, job_titles, experience_levels):
    # Define colors for different experience levels
    colors = ['blue', 'green', 'red', 'purple']

    for job_title_to_match in job_titles:
        st.subheader(f"Linear Regression Results for {job_title_to_match}")

        # Create a Plotly figure
        fig = px.line()
        fig.update_layout(
            xaxis_title='Work Year',
            yaxis_title='Salary in USD',
            title=f'Linear Regression for {job_title_to_match}',
        )

        for experience_level in experience_levels:
            if job_title_to_match == "Machine Learning Engineer" and experience_level == "EX":
                # Skip this specific combination
                continue

            model = models[job_title_to_match][experience_level]

            # Predict future results
            x_future = np.arange(2020, 2031).reshape(-1, 1)  # Extend the x-axis to 2030
            y_future = model.predict(x_future)

            # Add a trace for future predictions
            fig.add_trace(go.Scatter(
                x=x_future.ravel(),
                y=y_future,
                mode='lines',
                line=dict(color=colors[experience_levels.index(experience_level)]),
                name=f'{experience_level} - Future Prediction',
                hovertemplate='%{y:.2f} USD',
            ))

        # Customize the plot for the specific job title
        fig.update_layout(
            xaxis_title='Work Year',
            yaxis_title='Salary in USD',
            title=f'Linear Regression for {job_title_to_match}',
        )

        st.plotly_chart(fig)

# Plot regression results
plot_regression_results(trained_models, job_titles, experience_levels)

def predict_salaries_for_all_combinations(trained_models, job_titles, experience_levels, work_year_to_predict):
    predictions = {}  # To store predictions

    for job_title in job_titles:
        predictions[job_title] = {}  # Create a sub-dictionary for each job title

        for experience_level in experience_levels:
            model = trained_models.get(job_title, {}).get(experience_level)

            if model is not None:
                predicted_salary = model.predict([[work_year_to_predict]])[0]
                # Calculate the percentage difference from the salary in 2023
                original_salary_2023 = model.predict([[2023]])[0]
                percentage_difference = ((predicted_salary - original_salary_2023) / original_salary_2023) * 100
                predictions[job_title][experience_level] = (predicted_salary, percentage_difference)
            else:
                predictions[job_title][experience_level] = (None, None)

    return predictions

# Define the work year for which you want to make predictions
work_year_to_predict = 2030  # Replace with the desired work year

# Call the function to make predictions
predicted_salaries = predict_salaries_for_all_combinations(trained_models, job_titles, experience_levels, work_year_to_predict)

# Display the predicted salaries and percentage differences in Streamlit
for job_title in job_titles:
    st.subheader(f"Predicted Salaries for {job_title}")
    for experience_level in experience_levels:
        predicted_salary, percentage_difference = predicted_salaries[job_title][experience_level]
        if predicted_salary is not None:
            st.write(f'{experience_level} - Predicted salary for {job_title} in {work_year_to_predict}: {predicted_salary:.2f} USD')
            st.write(f'{experience_level} - Percentage difference from 2023: {percentage_difference:.2f}%')
        else:
            st.write(f'{experience_level} - Insufficient data for {job_title}.')
