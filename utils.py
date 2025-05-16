import numpy as np
import pandas as pd
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.special import beta as beta_func  # Beta function
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from dotenv import load_dotenv
import json
import warnings
warnings.filterwarnings('ignore')

def is_date_column(col):
    # Function to check if column names are dates
    try:
        pd.to_datetime(col)  # Try converting the column name to a date
        return True
    except:
        return False
    
def numeric_nonNumeric_col(df):
    '''
    seperates the numeric and non-numeric columns.
    '''
    return [col for col in df.columns if isinstance(col, int)], [col for col in df.columns if isinstance(col, str)]

def interpolate_full_range(df):
    # Extract numeric column names
    numeric_cols, nonNumeric_cols = numeric_nonNumeric_col(df)

    # Function to fill the row with the first valid value
    def fill_initial_nans(series):
        first_valid_idx = series.first_valid_index()
        last_valid_idx = series.last_valid_index()
        if first_valid_idx is not None:  # Ensure there's a valid index
            series.loc[:first_valid_idx] = series[first_valid_idx]   # Fill initial NaNs
        if last_valid_idx is not None:
            series.loc[last_valid_idx:] = series[last_valid_idx]
        return series

    # Apply the function 
    df[numeric_cols] = df[numeric_cols].apply(fill_initial_nans, axis=1)

    # Create full range of numeric columns from min to max
    full_range = np.arange(min(numeric_cols), max(numeric_cols) + 1)

    # Reindex DataFrame to include all missing columns
    df_numeric = df[numeric_cols].reindex(columns=full_range)

    # Interpolate missing values row-wise
    df_interpolated = df_numeric.interpolate(method='linear', axis=1)

    # Combine back with categorical columns
    return pd.concat([df_interpolated, df[nonNumeric_cols]], axis=1).reset_index(drop=True)

def find_daily_diff(df):
    numeric_cols, _ = numeric_nonNumeric_col(df)
    diff_daily = df[numeric_cols].copy()
    diff_daily.iloc[:, 1:] = diff_daily.iloc[:, 1:].values - diff_daily.iloc[:, :-1].values
    return diff_daily

# Define the logistic function
def logistic(x, L, k, x0):
    # Provide initial guesses for parameters L, k, x0
    return L / (1 + np.exp(-k * (x - x0)))

def logistic_fit(x,y):
    initial_guess = [8, 1, 4]
    # Fit the curve
    popt, _ = curve_fit(logistic, x, y, p0=initial_guess, bounds=(0, np.inf))
    # Use the fitted parameters to compute the model predictions
    y_fit = logistic(x, *popt)  
    return y_fit, popt

def beta_fit(x,y):
    # Define the scaled beta function for fitting
    def beta_function(x, A, alpha, beta):
         # Map x to t in [0,1]
        t = x / len(x)
        # Compute the beta
        return A * (t**(alpha-1)) * ((1-t)**(beta-1)) / beta_func(alpha, beta)

    # Provide initial guesses and bounds for parameters:
    # A is around the maximum y value, and alpha, beta > 1 for a unimodal curve that is zero at boundaries.
    initial_guess = [7, 3, 3]
    bounds = ([0, 1, 1], [20, 15, 15])

    # Fit the function
    popt, pcov = curve_fit(beta_function, x, y, p0=initial_guess, bounds=bounds)

    # Compute the fitted values
    y_fit = beta_function(x, *popt) 
    return y_fit

def gaussian_fit(x,y):
    # Define the Gaussian function
    def gauss(x, A, mu, sigma):
        return A * np.exp(-((x - mu)**2) / (2 * sigma**2))

    # Initial parameter guess: A (around max(y)), mu (center), sigma
    initial_guess = [np.max(y), np.median(x), 1]
    bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])

    # Fit the Gaussian function to the data
    popt, pcov = curve_fit(gauss, x, y, p0=initial_guess, bounds=bounds)

    # Compute fitted values
    y_fit = gauss(x, *popt)
    return y_fit

def calculate_fit_stats(original_values, fitted_values):
    # Compute statistical metrics
    mse = mean_squared_error(original_values, fitted_values) # Penalizes large errors
    rmse = np.sqrt(mse) # Easier to interpret (same unit as data)
    mae = mean_absolute_error(original_values, fitted_values) # Measures absolute errors
    r2 = r2_score(original_values, fitted_values) # Explains variance (0-1 range), closer to 1 is better
    nrmse = rmse / (np.max(original_values) - np.min(original_values)) # Typically, NRMSE < 0.1 is considered a good fit
    msd = np.mean((np.array(original_values) - np.array(fitted_values)) ** 2)
    # Return results as a dictionary
    return {
        "RMSE": float(round(rmse,2)),
        "MAE": float(round(mae,2)),
        "R2": float(round(r2,2)),
        "NRMSE": float(round(nrmse,2)),
        'MSD': float(round(msd,2))
    }

def BB_specifications(location,df_doy_cols,BB_percent=False):
    # return the proportion of bud break and median number of broken buds  
    if BB_percent:
        assumed_bb_percent = BB_percent
    elif location == "Kerikeri":
        assumed_bb_percent = .33
    elif location == "Te Puke":
        assumed_bb_percent = .5
    max_observed_buds = df_doy_cols.iloc[:,-1].median() / assumed_bb_percent

    # fit a sigmoid to budbreak observations
    first_doy = [col for col in df_doy_cols.columns if df_doy_cols[col].mean()>0][0]
    start_idx = df_doy_cols.columns.get_loc(first_doy)
    

    y_vals = df_doy_cols.iloc[:, start_idx:].median().values
    x_vals = range(len(y_vals))

    _, logistic_params = logistic_fit(x_vals, y_vals)
    bb_start_val = round(.05 * df_doy_cols.iloc[:,-1].median(),1)
    
    full_range_doy = np.arange(first_doy, df_doy_cols.columns[-1]+1,1)
    y_fit = logistic(range(0,len(full_range_doy)), *logistic_params) 
    BudBurstDOY = full_range_doy[y_fit > bb_start_val][0]

    PBB = [round(bud/df_doy_cols.iloc[:,-1].median(),2) for bud in df_doy_cols.median().values] 
    BB = [bud for bud in df_doy_cols.median().values]


    # try:
    #     valid_cols = [col for col in df_doy_cols.columns if df_doy_cols[col].median()  > round(0.05*max_observed_buds)]
    #     BudBurstDOY = valid_cols[0] if valid_cols else [col for col in df_doy_cols.columns if df_doy_cols[col].mean()  > round(0.05*max_observed_buds)]
    # except:
    #     print('*****************EMPTY DATAFRAME PASSED TO BB_specifications******************')
    # PBB = [round(bud/max_observed_buds,2) for bud in df_doy_cols.median().values] 
    # BB = [bud for bud in df_doy_cols.median().values]

    # print(location, ' :', BudBurstDOY, bb_start_val, df_doy_cols.iloc[:,-1].median())
    return PBB, BB, BudBurstDOY, max_observed_buds

def Flwr_specifications(MaxBB, df_doy_cols):
    valid_cols = [col for col in df_doy_cols.columns if df_doy_cols[col].median() > round(0.05 * MaxBB)]
    FlwrDOY = valid_cols[0] if valid_cols else [col for col in df_doy_cols.columns if df_doy_cols[col].mean() > round(0.05 * MaxBB)][0]
    return FlwrDOY


def split_phrase(phrase):
    if " " in phrase:
        return phrase.split(" ")  # Split by space
    elif "_" in phrase:
        return phrase.split("_")  # Split by underscore
    else:
        return [phrase, phrase] # repeat the treatmnet name for both the experiment and treatment
    
def seasonal_ave_fillna(df):

    # Define seasons based on the Southern Hemisphere
    seasons = {
        "Summer": list(range(355, 367)) + list(range(1, 80)),  # Dec 21 - Mar 20
        "Autumn": list(range(80, 172)),  # Mar 21 - Jun 20
        "Winter": list(range(172, 266)),  # Jun 21 - Sep 22
        "Spring": list(range(266, 355)),  # Sep 23 - Dec 20
    }

    # Compute seasonal averages
    seasonal_averages = {}
    for season, days in seasons.items():
        seasonal_averages[season] = df[df['doy'].isin(days)]['temp'].mean()

    # Fill missing values based on season
    def fill_missing(day):
        for season, days in seasons.items():
            if day in days:
                return seasonal_averages[season]

    df.loc[df['temp'].isna(), 'temp'] = df['doy'].apply(fill_missing)

    return df






#---------------------------------------------------------------------
#Function to create the default configuration for the model. This will be overridden as 
#required during experimentation
#---------------------------------------------------------------------
def base_model_config():
    model_config = {
            "StartDay" : '2000-05-1', # start accumulation of chill units (year selection does not matter here, it'll be turned into day of year)
            "Tc_chill": 15.9, # chill model
            "MinTemp": 7.786, # WangEngel model
            "OptTemp": 10, # WangEngel model
            "MaxTemp": 25.45, # WangEngel model
            "Tb_GDH": 8, # GDH model
            "Tu_GDH": 21, # GDH model
            "Tc_GDH": 25, # GDH model
            "ChillRequirement" : 1076.02966187,
            "HeatRequirement" : 999.34361384,
            "FlwrHeatRequirement" : 900,
            "InterpolationMethod": 'linear',
            "HeatAccFunc": 'WangEngel'


            }
    return model_config

