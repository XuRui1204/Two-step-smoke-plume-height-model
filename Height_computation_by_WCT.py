#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
This code contains:
1) reading the CALIPSO smoke extinction coefficient data from the gridded CALIPSO data based on the coordinates of the points where CALIPSO and MODIS overlap,
2) calculating the smoke height using the WCT method
'''


# In[ ]:


import numpy as np
import netCDF4 as nc
import os
import warnings
import time
from geopy import distance
from scipy.signal import find_peaks


def find_nearest_point(lat0,lon0,lines_map):
    '''
    this function returns the closest point lat and lon from lat0 and lon0 on the CALIPSO track
    '''
    lat,lon = np.where(lines_map==1)
    lat = -90+lat*0.1
    lon = -180+lon*0.1
    distance = (lon-lon0)**2+(lat-lat0)**2
    ind = np.argwhere(distance==distance.min())
    if ind.size>1:
        ind = ind[0]
    return lat[ind][0],lon[ind][0]


# ----------- main code ------------------
# elevation data read
f_elev = nc.Dataset('', 'r')
lon_elev = f_elev.variables['lon'][:]
lat_elev = f_elev.variables['lat'][:]
elev = f_elev.variables['elev'][:]
f_elev.close()

# CALIPSO&MODIS matched data read
smoke_extinc_532_w = np.empty((0, 11, 11, 545))
smoke_extinc_1064_w = np.empty((0, 11, 11, 545))
file_name = '' # CALIPSO&MODIS match data file
f_clps_m = nc.Dataset(file_name, 'r')
lon_mds = f_clps_m['lon'][:]
lat_mds = f_clps_m['lat'][:]
day_mds = f_clps_m['day'][:]

for imds in range(lat_mds.size):
    FILE_NAME = '' # CALIPSO grid file
    f_calipso = nc.Dataset(FILE_NAME, 'r')
    lon_calipso = f_calipso.variables['lon'][:]
    lat_calipso = f_calipso.variables['lat'][:]
    altitude_calipso = f_calipso.variables['altitude'][:]
    feature_type = f_calipso.variables['feature_type'][:]
    lines_map = f_calipso.variables['feature_type'][0,:,:,1]
    is_all_masked = np.ma.all(np.ma.getmaskarray(feature_type[0,:,:,:]), axis=2)
    lines_map[is_all_masked]=0
    lines_map[~is_all_masked]=1
    day_previous = iday

    lat0, lon0 = find_nearest_point(lat_mds[imds], lon_mds[imds], lines_map)
    ind_lon = np.argmin(np.abs(lon_calipso-lon0))
    ind_lat = np.argmin(np.abs(lat_calipso-lat0))
    if ind_lon + 6 >= lon_calipso.size:
        smk_ex_532_1 = f_calipso.variables['smoke_extinc_532'][0, ind_lat-5:ind_lat+6, ind_lon-5:3600, :]
        smk_ex_532_2 = f_calipso.variables['smoke_extinc_532'][0, ind_lat-5:ind_lat+6, 0:ind_lon+6-3600, :]
        smk_ex_532 = np.concatenate((smk_ex_532_1, smk_ex_532_2), axis=1)
    elif ind_lon - 5 < 0:
        smk_ex_532_1 = f_calipso.variables['smoke_extinc_532'][0, ind_lat-5:ind_lat+6, ind_lon-5+3600:3600, :]
        smk_ex_532_2 = f_calipso.variables['smoke_extinc_532'][0, ind_lat-5:ind_lat+6, 0:ind_lon+6, :]
        smk_ex_532 = np.concatenate((smk_ex_532_1, smk_ex_532_2), axis=1)
    else:
        smk_ex_532 = f_calipso.variables['smoke_extinc_532'][0, ind_lat-5:ind_lat+6, ind_lon-5:ind_lon+6, :]
    
    ind_lat_elev = np.argmin(np.abs(lat_elev-lat0))
    ind_lon_elev = np.argmin(np.abs(lon_elev-lon0))
    smk_extinc_532_lvlmean = np.ma.mean(smoke_extinc_532, axis=(1,2))
    if np.ma.notmasked_edges(smk_extinc_532_lvlmean) is None:
        h_calipso = np.nan
        continue
    ind_b, ind_t = np.ma.notmasked_edges(smk_extinc_532_lvlmean)
    smk_extinc_532_lvlmean[ind_t:ind_t+6].mask = False
    smk_extinc_532_lvlmean[ind_t:ind_t+6] = 0

    if np.abs(elev[ind_lat_elev, ind_lon_elev]-altitude[ind_b])>150:
        h_calipso = np.nan
        continue
    masked_value = np.sum(smk_extinc_532_lvlmean[ind_b:ind_t].mask)
    if masked_value/(ind_t-ind_b+1) > 0.4:
        h_calipso = np.nan
        continue
    
    # WCT
    n = 20
    b = np.arange(altitude[ind_b]-15, altitude[ind_t+6]+15, 50)
    W = np.zeros(b.size)
    a = n*30
    for i in range(b.size):
        ind1 = np.argmin(np.abs(altitude-(b[i]-a/2)))
        ind2 = np.argmin(np.abs(altitude-b[i]))
        ind3 = np.argmin(np.abs(altitude-(b[i]+a/2)))
        W[i] += np.sum(smk_extinc_532_lvlmean[ind1:ind2])
        W[i] -= np.sum(smk_extinc_532_lvlmean[ind2:ind3])
        W[i] = W[i]/n
    W = np.ma.masked_where(np.isnan(W),W)
    peaks, _ = find_peaks(W)
    peaks_above_threshold = [i for i in peaks if W[i] > 0.04]
    if peaks_above_threshold:
        last_peak_index = peaks_above_threshold[-1]
    else:
        h_calipso = np.nan
        continue
    h_calipso = b[last_peak_index]

