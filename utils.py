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

    # Apply the function row-wise
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


def logistic_fit(x,y):
    # Define the logistic function
    def logistic(x, L, k, x0):
        # Provide initial guesses for parameters L, k, x0
        return L / (1 + np.exp(-k * (x - x0)))
    initial_guess = [8, 1, 4]
    # Fit the curve
    popt, _ = curve_fit(logistic, x, y, p0=initial_guess, bounds=(0, np.inf))
    # Use the fitted parameters to compute the model predictions
    y_fit = logistic(x, *popt)  
    return y_fit

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
    nrmse = rmse / (original_values.max() - original_values.min()) # Typically, NRMSE < 0.1 is considered a good fit
    # Return results as a dictionary
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "NRMSE": nrmse
    }
