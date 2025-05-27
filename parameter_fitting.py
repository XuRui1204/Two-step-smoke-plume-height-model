#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
This code contains the parameter tuning process for different models.
blh and cape are from ERA5 hourly single level data,
bcaod is from EAC4 BCAOD @ 550nm data,
FRP is from MODIS FRP data,
lfc and bvf2 are calculated by Metpy.
'''


# In[ ]:


import numpy as np
import netCDF4 as nc
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from scipy.optimize import differential_evolution

# training set and testing set
X = np.column_stack((blh_clps, FRP_mean_clps, bcaod, cape_clps, elev_clps, lfc_clps_filled, theta_s, bvf2))
y = H_clps.copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sofiev original loss function
def obj_func_Sofiev(params):
    a, b, c, d = params
    JR = 0
    Nnan = 0
    for ifire in range(X_train.shape[0]):
        if np.ma.is_masked(X_train[ifire,1]) or np.isnan(X_train[ifire,1]):# FRP
            Nnan += 1
            continue
        if np.ma.is_masked(X_train[ifire,4]) or np.isnan(X_train[ifire,4]): # elev
            Nnan += 1
            continue
        if np.isnan(X_train[ifire, 7]) or np.ma.is_masked(X_train[ifire, 7]):
            Nnan += 1
            continue
        Hmdl = a*X_train[ifire,0] + b*((X_train[ifire,1]*1e6)/Pf0)**c / np.exp(d*X_train[ifire, 7]/N02) + X_train[ifire,4]
        if np.isnan(Hmdl):
            Nnan += 1
            continue
        bias = np.abs(y_train[ifire] - Hmdl)
        if bias > 150:
            JR += 1
    return JR

# RMSE loss function for Sofiev model
def obj_func_rmse(params):
    a, b, c, d = params
    RMSE = 0
    Nnan = 0
    for ifire in range(X_train.shape[0]):
        if np.ma.is_masked(X_train[ifire,1]) or np.isnan(X_train[ifire,1]):# FPR
            Nnan += 1
            continue
        if np.isnan(X_train[ifire, 7]) or np.ma.is_masked(X_train[ifire, 7]): # bvf2
            Nnan += 1
            continue
        Hmdl = a*X_train[ifire,0] + b*((X_train[ifire,1]*1e6)/Pf0)**c / np.exp(d*X_train[ifire,7]/N02) +X_train[ifire,4]        
        if np.isnan(Hmdl):
            Nnan += 1
            continue       
        bias = np.abs(y_train[ifire] - Hmdl)
        RMSE += bias**2
    RMSE = np.sqrt(RMSE / (y_train.size - Nnan)) if Nnan < y_train.size else np.inf
    return RMSE

# RMSE loss function for Hdetect
threshold_value = np.percentile(H_clps, 75)
def obj_func_detect(params):
    a, b, c, d, e = params
    Nnan = 0
    RMSE = 0
    for ifire in range(X_train.shape[0]):
        if y_train[ifire] < threshold_value:
            Nnan
            continue
        if np.ma.is_masked(X_train[ifire,1]) or np.isnan(X_train[ifire,1]):# FRP
            Nnan += 1
            continue
        if np.ma.is_masked(X_train[ifire,2]) or np.isnan(X_train[ifire,2]): # aod
            Nnan += 1
            continue
        if np.ma.is_masked(X_train[ifire,4]) or np.isnan(X_train[ifire,4]): # elev
            Nnan += 1
            continue
        H_detect_ifire = a * X_train[ifire,0] + b * ((X_train[ifire,1]*1e6)/Pf0 + d*X_train[ifire,2]/AOD0 + e*X_train[ifire,3]/CAPE0)**c +X_train[ifire,4]
        if np.isnan(H_detect_ifire):
            Nnan += 1
            continue
        bias = np.abs(y_train[ifire] - H_detect_ifire)
        RMSE += bias**2
    RMSE = np.sqrt(RMSE/((y_train.size-Nnan)))
    return RMSE

# RMSE loss function for two-step model
def obj_func_2step(params):
    a, b, c, d, e, w = params
    Nnan = 0
    RMSE = 0
    for ifire in range(X_train.shape[0]):
        if np.ma.is_masked(X_train[ifire,1]) or np.isnan(X_train[ifire,1]):# FRP
            Nnan += 1
            continue
        if np.ma.is_masked(X_train[ifire,2]) or np.isnan(X_train[ifire,2]): # aod
            Nnan += 1
            continue
        if np.ma.is_masked(X_train[ifire,4]) or np.isnan(X_train[ifire,4]): # elev
            Nnan += 1
            continue
        if np.isnan(X_train[ifire, 7]) or np.ma.is_masked(X_train[ifire, 7]): # bvf2
            Nnan += 1
            continue
            
        H_detect_ifire = detect_params[0] * X_train[ifire,0] + detect_params[1] * (X_train[ifire,1]*1e6/Pf0 + 
                                                                            detect_params[3]*X_train[ifire,2]/AOD0 + 
                                                                            detect_params[4]*X_train[ifire,3]/CAPE0)**detect_params[2] +X_train[ifire,4]
        if H_detect_ifire >= X_train[ifire,5] and not np.isnan(X_train[ifire,5]):
            Hmdl = a*X_train[ifire,0] + b*(X_train[ifire,1]*1e6/Pf0 + e*X_train[ifire,2]/AOD0 + w*X_train[ifire,3]/CAPE0)**c +X_train[ifire,4]
        else:
            Hmdl = a*X_train[ifire,0] + b*(X_train[ifire,1]*1e6/Pf0 + e*X_train[ifire,2]/AOD0)**c/np.exp(d*X_train[ifire,7]/N02) +X_train[ifire,4]
        bias = np.abs(y_train[ifire] - Hmdl)
        RMSE += bias**2
    RMSE = np.sqrt(RMSE/((y_train.size-Nnan)))
    return RMSE


# In[ ]:


# ------------------ main code -------------------
# Sofiev model + Original loss function
param_bounds = [(0, 1), (100, 1e4), (0.25, 0.5), (0.1, 1)]
result_sofiev = differential_evolution(
    obj_func_Sofiev,
    bounds=param_bounds,
    strategy='best1bin',
    maxiter=100,
    popsize=15,
    tol=0.01
)
optimal_params_sofiev = result_sofiev.x

# Sofiev model + RMSE loss function
initial_params = [0.24, 270, 0.35, 0.6]
param_bounds = [(0, 1), (100, 1e4), (0.25, 0.5), (0.1, 1)]
result_rmse = minimize(obj_func_rmse, initial_params, method='L-BFGS-B', bounds=param_bounds)
optimal_params = result_rmse.x

# H_detect + RMSE loss function
param_bounds = [(0, 1), (100, 1e4), (0.25, 0.5), (0,1e4), (0,1e4)]
initial_params = [0.24, 270, 0.35, 1, 1]
result = minimize(obj_func_detect, initial_params, method='L-BFGS-B', bounds=param_bounds)
detect_params = result.x

# Two-step model + RMSE loss function
initial_params = [0.24, 270, 0.35, 0.6, 1, 1]
param_bounds = [(0, 1), (0, 1e4), (0.25, 0.5), (0.1, 1), (0,1e4), (0,1e4)]
result_2step = minimize(obj_func_2step, initial_params, method='L-BFGS-B', bounds=param_bounds)
params_2step = result_2step.x

