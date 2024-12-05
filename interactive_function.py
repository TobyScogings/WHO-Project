#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split

# from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

# import statsmodels.api as sm
# import statsmodels.tools

# from statsmodels.stats.outliers_influence import variance_inflation_factor
import streamlit as st

st.title("Life Expectancy prediction app")
st.markdown("")

help = pd.read_csv("metadata.csv")
with st.expander("See here for a list of field descriptions"):
    st.dataframe(help)

prompt = "Do you want to run the full model (1) or run a censored model to cover sensitive data (2)?\n Select this for full model."
agree = st.checkbox(prompt)

st.subheader("Enter your particulars below")
if not agree:
    with st.form("model_2"):
        year = st.number_input("Year", min_value=2000, max_value=3000)
        adult_mortality = st.number_input("adult mortality rate", min_value=0)
        alcohol_consumption = st.number_input("alcohol consumption", min_value=0)
        hepatitis_b = st.number_input("hepatitis B immunization (%)", min_value=0, max_value=100)
        polio = st.number_input("polio immunization (%)", min_value=0, max_value=100)
        diphtheria = st.number_input("diphtheria immunization (%)", min_value=0, max_value=100)
        gdp = st.number_input("GDP", min_value=0)
        submit2 = st.form_submit_button("Submit info and predict life expectancy")
    
    if submit2:
        # placeholder for feature engineering, data transformation using the full model
        
        # placeholder for prediction using the full model
        st.subheader("Result:")
        predict2 = "holder for sensitive model"
        st.write(f"Your life expectancy is {predict2}.")

if agree:
    with st.form("model_1"):
        year = st.number_input("Year", min_value=2000, max_value=3000)

        u5_deaths = st.number_input("U5 Deaths per 1000", min_value=0, max_value=1000)
        adult_mortality = st.number_input("adult mortality rate", min_value=0)
        alcohol_consumption = st.number_input("alcohol consumption", min_value=0)
        hepatitis_b = st.number_input("hepatitis B immunization (%)", min_value=0, max_value=100)
        bmi = st.number_input("BMI", min_value=0, max_value=1000)
        polio = st.number_input("polio immunization (%)", min_value=0, max_value=100)
        diphtheria = st.number_input("diphtheria immunization (%)", min_value=0, max_value=100)
        hiv = st.number_input("HIV per 1000", min_value=0, max_value=1000)
        thinness_10_19 = st.number_input("thinness between 10-19", min_value=0, max_value=100)
        schooling = st.number_input("schooling years", min_value=0, max_value=100)
        economy_list = {0: "Developing", 1: "Developed"}
        economy_status = st.radio("economy status", options=economy_list.keys(), format_func=lambda x: economy_list.get(x), 
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
        # placeholder for feature engineering, data transformation using the full model

        # placeholder for prediction using the full model
        st.subheader("Result:")
        predict1 = "holder for full model"
        st.write(f"Your life expectancy is {predict1}.")

