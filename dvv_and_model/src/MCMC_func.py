import os
import h5py
import time
import emcee
import datetime
import numpy as np
import matplotlib as mpl
from scipy.integrate import quad
from multiprocessing import Pool, cpu_count
# For the speed up of integral with Low level calling function
import ctypes
from scipy import LowLevelCallable


def get_keys(modelcase):

    if modelcase.lower() == "base":

        keys = ['a0', 'p1', 'a_{precip}', 'p2', 't_{shiftdays}', 'log_f']

    elif modelcase.lower() == "evap":

        keys = ['a0', 'p1', 'a_{precip}', 'p2', 't_{shiftdays}', 'p3', 'log_f']

    else:
        print(f"{modelcase} is not defined. Please check the modelcase.\n")

    return keys


def get_init_param(**modelparam):
    """pos array"""
    nwalkers = modelparam["nwalkers"]
    keys = get_keys(modelparam["modelcase"])

    ndim = len(keys)
    pos = [modelparam[x][0] for x in keys] + 1e-4 * np.random.randn(nwalkers, ndim)

    return (pos, ndim, keys)


#--- Model ---#
#1. ---Precipitation and ground water level change---#
def SSW06(precip, phi, a, stackstep):
    """
    Modified from Tim Clements's Julia code
    Parameter 'a' has a unit of [1/(day_stackstep)]
    Parameter 'phi': constant porosity
    """
    Nprecip = len(precip)
    expij = [np.exp(-(a*stackstep)*x) for x in range(Nprecip)]
    GWL = np.convolve(expij, precip - np.mean(precip), mode='full')[:Nprecip] / phi
    return GWL


def compute_GWLchange(a_SSW06, **modelparam):

    GWL = SSW06(np.array(modelparam["precip"])/1e2, 0.05, a_SSW06, modelparam["averagestack_step"])
    GWL = GWL - np.mean(GWL)
    return GWL


#2. ---Temperature and thermoelastic change---#
def compute_tempshift(shift_hours, smooth_winlen = 6, **modelparam):
    """
    shifted and smooth the history of temperature
    shiftdays: days to shift
    smooth_winlen x shiftdays --> total smooth window
    """
    unix_tvec = modelparam["unix_tvec"]
    temperature = modelparam["CAVG"]  

    smooth_temp = np.convolve(temperature, np.ones(smooth_winlen)/smooth_winlen, mode='same')
    T_shift = np.interp(unix_tvec - shift_hours*1, unix_tvec, smooth_temp)

    return T_shift


def compute_tempshift_sur(shift_hours, smooth_winlen = 6, **modelparam):
    """
    shifted and smooth the history of temperature
    shiftdays: days to shift
    smooth_winlen x shiftdays --> total smooth window
    """
    unix_tvec = modelparam["unix_tvec"]
    temperature = modelparam["surface_temp"]  

    smooth_temp = np.convolve(temperature, np.ones(smooth_winlen)/smooth_winlen, mode='same')
    T_shift = np.interp(unix_tvec - shift_hours*1, unix_tvec, smooth_temp)

    return T_shift



#3. ---Linear trend---#
def compute_lineartrend(tvec, b):
    return b * (tvec-tvec[0])/1 # [1/sec]


#4.---Build---#
def model_base(theta, all=False, **modelparam):
    """no linear trend"""

    # parse the trial model parameters
    if modelparam["fixparam01"]:
        assert modelparam["ndim"] == len(theta)-3
    else:
        assert modelparam["ndim"] == len(theta)

    a0, p1, a_precip, p2, t_shiftdays, log_f= theta

    GWL_trim     = compute_GWLchange(a_precip, **modelparam)
    T_shift_trim = compute_tempshift(t_shiftdays, smooth_winlen = 6, **modelparam)

    # Construct model
    model = a0 + p1 * GWL_trim + p2 * T_shift_trim

    if all:
        return (model, p1 * GWL_trim, p2 * T_shift_trim)
    else:
        return model
    
    
def model_evap(theta, all=False, **modelparam):
    """no linear trend"""
    
    # parse the trial model parameters
    if modelparam["fixparam01"]:
        assert modelparam["ndim"] == len(theta)-3
    else:
        assert modelparam["ndim"] == len(theta)

    a0, p1, a_precip, p2, t_shiftdays, p3, log_f= theta

    GWL     = compute_GWLchange(a_precip, **modelparam)
    T_shift_solid = compute_tempshift(t_shiftdays, smooth_winlen = 6, **modelparam)
    T_shift_surface = compute_tempshift_sur(t_shiftdays, smooth_winlen = 6, **modelparam)

    # Construct model
    model = a0 + p1 * GWL + p2 * T_shift_solid + p3 * T_shift_surface

    if all:
        return (model, p1 * GWL_trim, p2 * T_shift_trim, p3 * T_shift_surface)
    else:
        return model
    

def model_wlin(theta, all=False, **modelparam):
    """linear trend"""

    if modelparam["fixparam01"]:
        assert modelparam["ndim"] == len(theta)-3
    else:
        assert modelparam["ndim"] == len(theta)

    a0, p1, a_precip, p2, t_shiftdays, b_lin, log_f = theta

    # get parameters from dictionary
    unix_tvec          = modelparam["unix_tvec"]

    GWL_trim     = compute_GWLchange(a_precip, **modelparam)
    T_shift_trim = compute_tempshift(t_shiftdays, smooth_winlen = 6, **modelparam)

    lintrend = compute_lineartrend(unix_tvec, b_lin)

    # Construct model
    model = a0 + p1 * GWL_trim + p2 * T_shift_trim + lintrend

    if all:
        return (model, p1 * GWL_trim, p2 * T_shift_trim, lintrend)
    else:
        return model

#--- posterior ---#
def log_likelihood(theta, **modelparam):
   

    dvv_data = modelparam["dvv_data_trim"]
    err_data = modelparam["err_data_trim"]
    
    coef = np.ones_like(err_data, dtype=np.float32)
    coef[err_data > 1.5] = 0.0

    modelcase = modelparam["modelcase"]
    keys = modelparam["keys"]

    log_f = theta[keys.index("log_f")]

    if modelcase.lower() == "base":
        model = model_base(theta, all=False, **modelparam)
    elif modelcase.lower() == "evap":
        model = model_evap(theta, all=False, **modelparam)
        
    cc = np.nansum(np.multiply(dvv_data, model))
    n1 = np.nansum(np.square(dvv_data))
    n2 = np.nansum(np.square(model))
    cc = cc / np.sqrt(n1 * n2)

#     sigma2 = err_data ** 2
#     sigma2 = err_data ** 2 + np.exp(2 * log_f)
    sigma2 = err_data ** 2 + model ** 2 * np.exp(2 * log_f)

#     return -0.5 * np.nansum((dvv_data - model) ** 2 / sigma2 + np.log(sigma2)) # 2pi is ignored
#     return -0.5 * np.nansum((dvv_data - model) ** 2 / sigma2 + np.log(sigma2*2*np.pi)) + (cc - 1) * np.nansum(sigma2)
#     return -0.5 * np.nansum((dvv_data - model) ** 2 / sigma2) + (cc - 1) * np.nansum(sigma2)

#     return -0.5 * np.nansum((dvv_data - model) ** 2 / sigma2 * coef + np.log(sigma2*2*np.pi)) + (cc - 1) * np.nansum(sigma2) * 5
    return -0.5 * np.nansum((dvv_data - model) ** 2 / sigma2 * coef + np.log(sigma2*2*np.pi))



def log_prior(theta, **modelparam):

    keys = modelparam["keys"]
    for i, key in enumerate(keys):
        vmin, vmax = modelparam[key][1]
        if (theta[i] < vmin) or (vmax < theta[i]):
            return -np.inf

    return 0

def log_probability(theta0, **modelparam):

    if modelparam["fixparam01"] == True:

        if modelparam["modelcase"]=="base":
            theta = np.concatenate((theta0[0:2], [modelparam["a_{precip}_fixed"]], theta0[2:5]), axis=None)
        else:
            theta = np.concatenate((theta0[0:2], [modelparam["a_{precip}_fixed"]], theta0[2:6]), axis=None)
    else:
        theta = theta0
    lp = log_prior(theta, **modelparam)

    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, **modelparam)



def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def compute_AIC(y_obs, y_syn, k):
    assert len(y_obs) == len(y_syn)
    N = len(y_obs) - np.count_nonzero(np.isnan(y_obs))
    return N*np.log((1/N)*np.nansum((y_obs - y_syn)**2)) + 2*k

def compute_BIC(y_obs, y_syn, k):
    assert len(y_obs) == len(y_syn)
    N = len(y_obs) - np.count_nonzero(np.isnan(y_obs))
    return N*np.log((1/N)*np.nansum((y_obs - y_syn)**2)) + k*np.log(N)


