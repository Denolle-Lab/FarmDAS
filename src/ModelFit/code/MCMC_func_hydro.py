import datetime
import os
import time

import numpy as np
import h5py

from scipy.integrate import quad

# For the speed up of integral with Low level calling function
import ctypes
from scipy import LowLevelCallable

import matplotlib as mpl

import emcee # MCMC sampler
# import corner

from multiprocessing import Pool, cpu_count


# ---Initialization---#
def get_keys(modelcase):
    if modelcase.lower() == "dv_temp":
        keys = ['a0', 'p4', 'p2', 't_{shiftdays}', 'b_{lin}', 'logf']  # dv + temperature
    elif modelcase.lower() == "temp":
        keys = ['a0',  'p2', 't_{shiftdays}', 'b_{lin}', 'logf']  # temperature
    elif modelcase.lower() == "dv":
        keys = ['a0', 'p4', 'b_{lin}', 'logf']  # dv
    else:
        print(f"{modelcase} is not defined. Please check the modelcase.\n")
        exit(1)

    return keys


def get_init_param(**modelparam):
    """
    return the pos array for the initial chains of MCMC
    """
    
    keys = get_keys(modelparam["modelcase"])

    nwalkers = modelparam["nwalkers"]

    ndim = len(keys)
    pos = [modelparam[x][0] for x in keys] + 1e-4 * np.random.randn(nwalkers, ndim)

    return (pos, ndim, keys)


# ---Model Components---#
# 1. ---Soil moisture to ground water level change--- #
def compute_soil2GWL(**modelparam):
    """
    mimic the GWL change with soil moisture equivalent water thickness
    """
    # print(soilmois, modelparam["soil"])
    fitting_period_ind = modelparam["fitting_period_ind"]
    GWL = np.array(modelparam["soil"])
    return GWL[fitting_period_ind]


# 2. ---Temperature and thermoelastic change---#
def compute_tempshift(shift_days, smooth_winlen=6, **modelparam):
    """
    compute the shifted time history of temperature with smoothing
    Input:
    t_shiftdays: days to shift the time history of temperature
    smooth_winlen: datapoints of smoothing. e.g. given 15 days sliding window, the winlen is 6 * 15 days = 3 months.
    """
    unix_tvec = modelparam["unix_tvec"]
    fitting_period_ind = modelparam["fitting_period_ind"]
    temperature = modelparam["CAVG"] # temperature in degree Celsius
    # print(shift_days, modelparam["t_shiftdays"])

    smooth_temp = np.convolve(temperature, np.ones(smooth_winlen)/smooth_winlen, mode='same')
    T_shift = np.interp(unix_tvec - shift_days*86400, unix_tvec, smooth_temp)
    return T_shift[fitting_period_ind]


# 3. ---Linear trend---#
def compute_lineartrend(tvec, b):
    """
    return linear trend with the slope of b, whose unit is [1/day].
    """
    return b * (tvec-tvec[0])/86400.0  # [1/day]


# ---Make model---#
def model_dv_temp(theta, all=False, **modelparam):
    """
    dv/v and temp model with linear trend term. --> to model hydro-term
    """
    assert modelparam["ndim"] == len(theta)

    a0, p4, p2, t_shiftdays, b_lin, log_f = theta  
    # get parameters from dictionary
    unix_tvec = modelparam["unix_tvec"]
    fitting_period_ind = modelparam["fitting_period_ind"]
    dv_trim = modelparam["dv"]
    T_shift_trim = compute_tempshift(t_shiftdays, smooth_winlen=6, **modelparam)

    # -------------------------------------------------------------------#
    lintrend = compute_lineartrend(unix_tvec[fitting_period_ind], b_lin)
    # Construct model
    model = a0 + p4 * dv_trim + p2 * T_shift_trim + lintrend
    return model


def model_temp(theta, all=False, **modelparam):
    """
    temp model with linear trend term. --> to model hydro-term
    """
    assert modelparam["ndim"] == len(theta)

    a0, p2, t_shiftdays, b_lin, log_f = theta
    # get parameters from dictionary
    unix_tvec = modelparam["unix_tvec"]
    fitting_period_ind = modelparam["fitting_period_ind"]
    T_shift_trim = compute_tempshift(t_shiftdays, smooth_winlen=6, **modelparam)

    # -------------------------------------------------------------------#
    lintrend = compute_lineartrend(unix_tvec[fitting_period_ind], b_lin)

    # Construct model
    model = a0 + p2 * T_shift_trim + lintrend
    return model


def model_dv(theta, all=False, **modelparam):
    """
    dv/v model with linear trend term. --> to model hydro-term
    """
    assert modelparam["ndim"] == len(theta)

    a0, p4, b_lin, log_f = theta
    # get parameters from dictionary
    unix_tvec = modelparam["unix_tvec"]
    fitting_period_ind = modelparam["fitting_period_ind"]
    dv_trim = modelparam["dv"]

    # ------------------------------------------------------------------#
    lintrend = compute_lineartrend(unix_tvec[fitting_period_ind], b_lin)

    # Construct model
    model = a0 + p4 * dv_trim + lintrend
    return model


# ---Log probabilities---#
# def log_likelihood(theta, unix_tvec, y_trim, yerr_trim, precip, temperature, fitting_period_ind, unix_tSS, unix_tPF):
def log_likelihood(theta, **modelparam):
    """
    Note: precip and temperature should have same length and timing with unix_tvec
    """
    # parse parameters
    data_trim = modelparam["soil"]
    # dvv_data_trim  =modelparam["dvv_data_trim"]
    # err_data_trim  =modelparam["err_data_trim"]

    modelcase = modelparam["modelcase"]
    keys = modelparam["keys"]

    log_f = theta[keys.index("logf")]

    # ---Select model---#
    if modelcase.lower() == "dv_temp":
        model = model_dv_temp(theta, all=False, **modelparam)
    elif modelcase.lower() == "temp":
        model = model_temp(theta, all=False, **modelparam)
    elif modelcase.lower() == "dv":
        model = model_dv(theta, all=False, **modelparam)

    # sigma2 = yerr_trim ** 2 + model ** 2 * np.exp(2 * log_f)
    # sigma2 = err_data_trim ** 2 + np.exp(2 * log_f) # 2022.2.21 Applying constant over/under estimation in error
    sigma2 = np.exp(2 * log_f)

    return -0.5 * np.nansum((data_trim - model) ** 2 / sigma2 + np.log(sigma2))  # 2pi is ignored


# assign boundary of parammeters as prior probability
def log_prior(theta, **modelparam):
    # print(theta)
    # print(modelparam)
    keys = modelparam["keys"]

    # 2. evaluate the boundaries
    for i, key in enumerate(keys):
        vmin, vmax = modelparam[key][1]
        val = theta[i]
        # print(key, val)

        if (val < vmin) or (vmax < val):
            return -np.inf      
    # if all the trial parameters are within the boundaries, return 0.
    return 0


def log_probability(theta0, **modelparam):
    time.process_time()
    # print(type(modelparam))
    # 2023.04.07 Update: add 'fixparam01' flag to fix the aprecip, log10tmin1 and log10tmin2
    # print(modelparam.keys())
    # if modelparam["fixparam01"] == True;
    #    # fix the aprecip, log10tmin1 and log10tmin2
    #    if  modelparam["modelcase"]=="base":
    #        theta = np.concatenate((theta0[0:2], [modelparam["a_{precip}_fixed"]], theta0[2:5], [modelparam["log10tmin1_fixed"]],
    #                          theta0[5:7], [modelparam["log10tmin2_fixed"]], theta0[7:9]), axis=None)
    #    elif modelparam["modelcase"]=="wlin":
    #        theta = np.concatenate((theta0[0:2], [modelparam["a_{precip}_fixed"]], theta0[2:5], [modelparam["log10tmin1_fixed"]],
    #                          theta0[5:7], [modelparam["log10tmin2_fixed"]], theta0[7:10]), axis=None)
    # else:
        # do not fix the parameters
    theta = theta0

    lp = log_prior(theta, **modelparam)
    # print(lp)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, **modelparam)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


def compute_AIC(y_obs, y_syn, k):
    assert len(y_obs) == len(y_syn)
    N = len(y_obs)
    return N*np.log((1/N)*np.nansum((y_obs - y_syn)**2)) + 2*k


def compute_BIC(y_obs, y_syn, k):
    assert len(y_obs) == len(y_syn)
    N = len(y_obs)
    return N*np.log((1/N)*np.nansum((y_obs - y_syn)**2)) + k*np.log(N)
