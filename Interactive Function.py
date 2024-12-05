#!/usr/bin/env python
# coding: utf-8

# <h4>Data and Module Importing</h4>

# In[31]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

def predict(features, dv, model):
    '''
    Transforms provided features dictionary and predicts if patient has heart disease
    Return the predicted probability of having heart disease in range [0,1].
    '''
    transformed = dv.transform(features)
    y_pred = model.predict_proba(transformed)[:,1] >= 0.5
    return y_pred
    



# def model_selection():

#     # Error Handling
    
#     try: model_choice = int(input("""Do you want to run the full model (1) or run a censored model to cover sensitive data (2)?
#     Enter your option here: """))
#     except:
#         print("Invalid input. Please enter either 1 or 2 to choose your model")
#         model_selection()

#     if model_choice == 1:

#     # Model FE and defining stage
    
#         X_train_fe = feature_eng_full(X_train)
#         model_cols = X_train_fe.columns

#     # Model Metrics
#         global model_state
#         model_state = "full"
#         modelling(model_cols)
#         print() # Line Break
#         model_state = "VIF optimised"
#         modelling(optimal_cols)

#     elif model_choice == 2:
#         print("This is a placeholder for the sensitive model")
#     else:
#         print("This is not one of the options. Please enter either 1 or 2 to choose your model")
#         model_selection()


# # In[140]:


# def feature_eng_full(data):
#     data = data.copy()

#     # Removing autocorrelated columns
    
#     data = data.drop(columns = ['Country', 'Economy_status_Developing', 'Infant_deaths'])
    
#     # One hot encoding
    
#     data = pd.get_dummies(data, columns = ['Region'], drop_first = True, prefix = 'Region', dtype=int) 

#     # Fixing exponential relationship

#     data['log_GDP'] =  np.log(data['GDP_per_capita'])

#     # Scaling
    
#     scaler = StandardScaler()
#     data[data.columns] = scaler.fit_transform(data[data.columns])

#     # Removing columns we are not interested in for our model

#     data = data.drop(columns = ['Measles', 'GDP_per_capita', 'Population_mln', 'Thinness_five_nine_years'])
    
#     # VIF

#     data_col = data.columns
    
#     calculate_vif(data[data_col])
    
#     data = sm.add_constant(data)
#     return data


# # In[19]:


# def calculate_vif(X, thresh = 5.0):
#     variables = list(range(X.shape[1]))
#     dropped = True
#     while dropped:
#         dropped = False
#         # this bit uses list comprehension to gather all the VIF values of the different variables
#         vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
#                for ix in range(X.iloc[:, variables].shape[1])]
        
#         maxloc = vif.index(max(vif)) # getting the index of the highest VIF value
#         if max(vif) > thresh:
#             del variables[maxloc] # we delete the highest VIF value on condition that it's higher than the threshold
#             dropped = True # if we deleted anything, we set the 'dropped' value to True to stay in the while loop    
    
#     global optimal_cols 
#     optimal_cols = list(X.columns[variables])
#     optimal_cols.append('const')

#     # We now create a global variable and assign the list of columns still in the valid set to it, remembering to add the constant back in. We can use this to check for an optimal condition number.
    
#     return optimal_cols


# # In[122]:


# def modelling(col):

#     # Modelling Stage
    
#     lin_reg = sm.OLS(y_train, X_train_fe[col])
#     results = lin_reg.fit()

#     # Metrics Observations 
    
#     print(f"\nThe following shows the level of success our {1} model has with predicting life expectancy:\n")
#     print(f"""
# P-Values:

# {round(results.pvalues,3)}

# R-Squared:
    
# {results.rsquared}
    
# AIC and BIC:
    
# {results.aic}
# {results.bic}
    
# Condition Number:
    
# {results.condition_number}
# """)

#     # RMSE Calculations
    
#     y_pred = results.predict(X_train_fe[col])
#     rmse = statsmodels.tools.eval_measures.rmse(y_train, y_pred)
#     print(f"RMSE:\n\n{rmse}")
#     # print(results.summary())


# # In[211]:


# def user_inputs_1():
#     user_values = ['year', 'U5 Deaths per 1000', 'adult mortality rate',
#        'alcohol consumption', 'hepatitis B immunization (%)', 'BMI', 'polio immunization (%)',
#        'diphtheria immunization (%)', 'HIV per 1000', 
#        'thinness between 10-19', 'schooling years',
#        'economy status (Developed or Developing)', 'region', 'GDP']

#     user_dict = {}

#     for each in user_values:
#         if each in ['economy status (Developed or Developing)', 'region']:
#             user_input = input(f"Please enter your value for {each}: ")
#             user_dict[each] = user_input
#         else:
#             while True:
#                 try:
#                     user_input = int(input(f"Please enter your value for {each}: "))
#                     user_dict[each] = user_input
#                     break
#                 except:
#                     print("This must be an integer, try again")
            
            
#     for a, b in user_dict.items():
#         print(f"{a.title()}: {b}")


# # In[213]:


# user_inputs_1()

