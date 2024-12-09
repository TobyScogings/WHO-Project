#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st

path = "Life Expectancy Data.csv"
metapath = "metadata.csv"

#------------------------------------------------------#
#                   DEFINE FUNCTIONS
#------------------------------------------------------#
# Load training data
def get_training_data(path):
    df = pd.read_csv(path)
    feature_cols = list(df.columns)
    feature_cols.remove('Life_expectancy')

    X = df[feature_cols]
    y = df['Life_expectancy']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)
    return X_train

# Modified from Toby's "Interactive Function.ipynb" > feature_eng_full()
# to return the StandardScaler generated from training data
def feature_eng_full(data):
    data = data.copy()
    
    # One hot encoding
    data = pd.get_dummies(data, columns = ['Region'], drop_first = True, prefix = 'Region', dtype=int) 

    # Fixing exponential relationship
    data['log_GDP'] =  np.log(data['GDP_per_capita'])

    # Removing autocorrelated columns and unwanted columns     
    data = data.drop(columns = ['Country', 'Economy_status_Developing', 'Infant_deaths', 'GDP_per_capita',
                                'Measles', 'Population_mln', 'Thinness_five_nine_years'])
    
    # Scaling
    scaler_cols = data.columns
    scaler = StandardScaler()
    data[scaler_cols] = scaler.fit_transform(data[scaler_cols])
    return scaler, scaler_cols

def transform_inputs(scaler, inputs):
    # reshape to (1,21) before rescale
    arr = np.array(list(inputs.values()) ).reshape(1,-1)
    arr_out = scaler.transform(arr)
    transformed = dict(zip(inputs.keys(), arr_out.flatten().tolist()))
    return transformed

# Coefficients retrieved from Toby's "Interactive Function.ipynb" > modelling()
def predict_full_model(inputs):
    full_coef = {'const': 68.868, 'Year': 0.1834, 'Under_five_deaths': -3.4189, 
                   'Adult_mortality': -5.3247, 'Alcohol_consumption': -0.089, 
                   'Hepatitis_B': -0.1266, 'BMI': -0.4056, 'Polio': 0.3417, 'Diphtheria': -0.228, 
                   'Incidents_HIV': 0.1931, 'Thinness_ten_nineteen_years': -0.1251, 'Schooling': 0.2649, 
                   'Economy_status_Developed': 1.0693, 'Region_Asia': 0.0844, 
                   'Region_Central America and Caribbean': 0.5687, 'Region_European Union': -0.2343, 
                   'Region_Middle East': 0.0585, 'Region_North America': 0.0846, 
                   'Region_Oceania': -0.2057, 'Region_Rest of Europe': 0.1471, 'Region_South America': 0.3875, 
                   'log_GDP': 0.652}
    
    result = full_coef['const'] + \
             (inputs["Year"] * full_coef['Year']) + (inputs['Under_five_deaths'] * full_coef['Under_five_deaths']) + \
             (inputs['Adult_mortality'] * full_coef['Adult_mortality']) + (inputs['Alcohol_consumption'] * full_coef['Alcohol_consumption']) + \
             (inputs['Hepatitis_B'] * full_coef['Hepatitis_B']) + (inputs['BMI'] * full_coef['BMI']) + \
             (inputs['Polio'] * full_coef['Polio']) + (inputs['Diphtheria'] * full_coef['Diphtheria']) + \
             (inputs['Incidents_HIV'] * full_coef['Incidents_HIV']) + (inputs['Thinness_ten_nineteen_years'] * full_coef['Thinness_ten_nineteen_years'])+ \
             (inputs['Schooling'] * full_coef['Schooling']) + (inputs["Economy_status_Developed"] * full_coef["Economy_status_Developed"])+ \
             (inputs["log_GDP"] * full_coef["log_GDP"]) + full_coef[f"Region_{region}"]             
    return result

# Coefficients retrieved from Vivien's "eda_modeling_vs.ipynb" > Modelling (minimalistic) > Seventh model
def predict_censored_model(inputs):
    minimal_coef = {'const': 68.8680,
                    'Year':	0.3297,
                    'Alcohol_consumption': 0.7905,
                    'Adult_mortality': -7.1659,
                    'log_GDP': 2.1598,
                    }
    result = minimal_coef['const'] + (inputs["Year"] * minimal_coef['Year']) + \
             (inputs['Adult_mortality'] * minimal_coef['Adult_mortality']) + \
             (inputs['Alcohol_consumption'] * minimal_coef['Alcohol_consumption']) + \
             (inputs["log_GDP"] * minimal_coef["log_GDP"])
    return result

#------------------------------------------------------#
#       PRELOAD FITTED STANDARDSCALER
#------------------------------------------------------#
# Load training data to fit a StandardScaler
data = get_training_data(path)
scaler, scaler_cols = feature_eng_full(data)
# print the columns required to pass into fitted StandardScaler
print(scaler_cols)

#------------------------------------------------------#
#                   STREAMLIT FRONTEND
#------------------------------------------------------#
st.image("app_header_image.png", 
         caption="World Health Organisation logo",
         use_container_width=True)
# st.title("Life Expectancy prediction app")
# st.markdown("")

help = pd.read_csv(metapath)
with st.expander("See here for a list of field descriptions"):
    st.dataframe(help)

prompt = "Do you want to run the full model (1) or run a censored model to cover sensitive data (2)?\n Select this for full model."
agree = st.checkbox(prompt)

st.subheader("Enter your particulars below")
# Uses censored model to exclude sensitive data
if not agree:
    with st.form("model_2"):
        year = st.number_input("Year", min_value=2000, max_value=3000)
        adult_mortality = st.number_input("Adult Mortality Rate", min_value=0)
        alcohol_consumption = st.number_input("Alcohol Consumption", min_value=0)
        # hepatitis_b = st.number_input("hepatitis B immunization (%)", min_value=0, max_value=100)
        # polio = st.number_input("polio immunization (%)", min_value=0, max_value=100)
        # diphtheria = st.number_input("diphtheria immunization (%)", min_value=0, max_value=100)
        gdp = st.number_input("GDP", min_value=0)
        submit2 = st.form_submit_button("Submit info and predict life expectancy")
    
    if submit2:
        # feature engineering, data transformation using the censored model
        gdp = np.log(gdp)
        inputs = {'Year':year, 'Under_five_deaths': 0, 'Adult_mortality':adult_mortality, 
                 'Alcohol_consumption': alcohol_consumption, 'Hepatitis_B': 0, 
                 'BMI': 0, 'Polio': 0, 'Diphtheria': 0, 'Incidents_HIV': 0,
                'Thinness_ten_nineteen_years': 0,
                'Schooling': 0, 'Economy_status_Developed': 0,
                'Region_Asia':0, 'Region_Central America and Caribbean':0,
                'Region_European Union':0, 'Region_Middle East':0, 'Region_North America':0,
                'Region_Oceania':0, 'Region_Rest of Europe':0, 'Region_South America':0,
                'log_GDP': gdp
                }
        transformed_inputs = transform_inputs(scaler, inputs)
        print(transformed_inputs)

        # prediction using the censored model
        predict2 = predict_censored_model(transformed_inputs)
        st.subheader("Result:")
        st.write(f"Your life expectancy is {predict2 :.1f} years.")

# Uses full model to include sensitive data
if agree:
    with st.form("model_1"):
        year = st.number_input("Year", min_value=2000, max_value=3000)

        u5_deaths = st.number_input("Under 5 Deaths per 1000", min_value=0, max_value=1000)
        adult_mortality = st.number_input("Adult Mortality Rate", min_value=0)
        alcohol_consumption = st.number_input("Alcohol Consumption", min_value=0)
        hepatitis_b = st.number_input("Hepatitis B Immunization (%)", min_value=0, max_value=100)
        bmi = st.number_input("BMI", min_value=0, max_value=1000)
        polio = st.number_input("Polio Immunization (%)", min_value=0, max_value=100)
        diphtheria = st.number_input("Diphtheria Immunization (%)", min_value=0, max_value=100)
        hiv = st.number_input("HIV per 1000", min_value=0, max_value=1000)
        thinness_10_19 = st.number_input("Thinness between 10-19 years old (%)", min_value=0, max_value=100)
        schooling = st.number_input("Schooling Years", min_value=0, max_value=20)
        economy_list = {0: "Developing", 1: "Developed"}
        economy_status = st.radio("Economy Status", options=economy_list.keys(), format_func=lambda x: economy_list.get(x), 
                    index=None)
        region = st.selectbox(
                    "Region",
                    ("Asia", "Central America and Caribbean", "European Union",
                    "Middle East", "North America", "Oceania", "Rest of Europe", "South America"),
                    index=None,
                    placeholder="pick a region",
                )
        gdp = st.number_input("GDP", min_value=0)

        submit1 = st.form_submit_button("Submit info and predict life expectancy")
    
    if submit1:
        # feature engineering, data transformation using the full model
        gdp = np.log(gdp)
        inputs = {'Year':year, 'Under_five_deaths': u5_deaths, 'Adult_mortality':adult_mortality, 
                 'Alcohol_consumption': alcohol_consumption, 'Hepatitis_B': hepatitis_b, 
                 'BMI': bmi, 'Polio': polio, 'Diphtheria': diphtheria, 'Incidents_HIV': hiv,
                'Thinness_ten_nineteen_years': thinness_10_19,
                'Schooling': schooling, 'Economy_status_Developed': economy_status,
                'Region_Asia':0, 'Region_Central America and Caribbean':0,
                'Region_European Union':0, 'Region_Middle East':0, 'Region_North America':0,
                'Region_Oceania':0, 'Region_Rest of Europe':0, 'Region_South America':0,
                'log_GDP': gdp
                }
        inputs[f"Region_{region}"] = 1
        transformed_inputs = transform_inputs(scaler, inputs)
        print(transformed_inputs)

        # prediction using the full model
        predict1 = predict_full_model(transformed_inputs)
        st.subheader("Result:")
        st.write(f"Your life expectancy is {predict1 :.1f} years.")

