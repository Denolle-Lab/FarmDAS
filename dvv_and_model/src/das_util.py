import sys
sys.path.append(".")
sys.path.append("noisepy4das_repo/NoisePy4DAS-SeaDAS/src")
sys.path.append("noisepy4das_repo/NoisePy4DAS-SeaDAS/DASstore")


import os
import h5py
import math
import time

import numpy as np
import pandas as pd
import matplotlib

from tqdm import tqdm
from obspy import UTCDateTime
from datetime import datetime
from datetime import timedelta
from functools import partial
from scipy.signal import butter
from scipy.signal import detrend
from scipy.signal import decimate
from scipy.signal import filtfilt
from scipy.signal import spectrogram
from dasstore.zarr import Client
from multiprocessing import Pool
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

def read_decimate(file_path, dsamp_factor=20, start_ch=0, end_ch=100):
    with h5py.File(file_path,'r') as f:      
        minute_data = f['Acquisition']['Raw[0]']['RawData'][:, start_ch:end_ch].T
    downsample_data = decimate(minute_data, q=dsamp_factor, ftype='fir', zero_phase=True)   
    return downsample_data


# %% extract time from file name (by Ethan Williams)
def get_tstamp(fname):
    datestr = fname.split('_')[1].split('-')
    y = int(datestr[0])
    m = int(datestr[1])
    d = int(datestr[2])
    timestr = fname.split('_')[2].split('.')
    H = int(timestr[0])
    M = int(timestr[1])
    S = int(timestr[2])
    return UTCDateTime('%04d-%02d-%02dT%02d:%02d:%02d' % (y,m,d,H,M,S))

def get_tstamp_dts(time_label):
    datestr = time_label.split(' ')[0].split('/')
    timestr = time_label.split(' ')[1]
    y = int(datestr[0])
    m = int(datestr[1])
    d = int(datestr[2])
    return UTCDateTime('%04d-%02d-%02dT%08s' % (y,m,d,timestr))

# %% calculate the NFFT for spectrogram (by Dominik Gräff)
def calc_NFFT(trace, sample_rate, power_of_2=True):
    '''calculate meaningful number of fourier samples for spectrogram'''
    NFFT = int(trace.shape[-1]/1000) # results in ~1000 bins
    if power_of_2:
        NFFT = int(2**(math.floor(math.log(NFFT, 2)))) # power of 2 < than original value
    print(r'NFFT={} samples, equivalent to {} seconds'.format(NFFT, NFFT/sample_rate))
    return NFFT, NFFT/sample_rate


# %% use pandas to find the best time-ticks
def x_tick_locs(stime, etime):
    '''calculate where to put x-ticks'''
    fmtfreq_list = {'years':['1YS'],
                    'months':['6MS','1MS'],
                    'days':['10d','5d','1d'],
                    'hours':['12h','6h','3h','1h'],
                    'minutes':['30min','15min','10min','5min','1min'],
                    'seconds':['30s','15s','10s','5s','1s']}

    for key in fmtfreq_list.keys():
        for value in fmtfreq_list[key]:
            daterange = pd.date_range(stime, etime+pd.Timedelta('1d'), 
                          freq=value, normalize=True)
            daterange = [t for t in daterange if t>=stime if t<=etime]
            if len(daterange)<6:
                continue
            else:
                return key, daterange

def x_labels_fmt(key, same_superior):
    '''x-ticks and axis formatting'''
    # if no change of superior unit
    if same_superior:
        fmtlabels_list = {'years':('%Y', ('', '', '[Year]')),
                          'months':('%b', ('of', '%Y', '[Month]')),
                          'days':('%-d %b', ('of', '%Y' '[Day Month]')),
                          'hours':('%H:%M', ('of', '%-d %b %Y', '[Hour:Minute]')),
                          'minutes':('%H:%M', ('of', '%-d %b %Y', '[Hour:Minute]')),
                          'seconds':('%H:%M:%S', ('of', '%-d %b %Y', '[Hour:Minute:Second]'))}
    # if superior unit changes
    if not same_superior:
        fmtlabels_list = {'years':('%Y', ('', '', '[Year]')),
                          'months':('%b %Y', ('', '', '[Month Year]')),
                          'days':('%-d %b', ('in', '%Y', '[Day Month]')),
                          'hours':('%-d %b %H:%M', ('in', '%Y', '[Day Month  Hour:Minute]')),
                          'minutes':('%H:%M', ('of', '%-d %b %Y', '[Hour:Minute]')),
                          'seconds':('%H:%M:%S', ('of', '%-d %b %Y', '[Hour:Minute:Second]'))}
    return fmtlabels_list[key]

def t_array(t):
    '''returns np.array([year,month,day,hour,minute,second])'''
    t_arr = [t.year,t.month,t.day,t.hour,t.minute,t.second]
    return t_arr

def translate_daterange_intervals(daterange, t_keys):
    '''find daterange spacing time unit'''
    t_arrs = [t_array(t) for t in daterange]
    for i, key in enumerate(t_keys):
        if not t_arrs[0][:i+1]==t_arrs[1][:i+1]:
            key = t_keys[i]
            return key
        
def nice_x_axis(stats, bins, t_int=False):
    stime = stats.starttime.datetime
    etime = stats.endtime.datetime
    t_bins = [(stats.starttime+t).datetime for t in bins]

    # units into which humans subdivide time
    t_keys = ['years','months','days','hours','minutes','seconds']

    # if fixed x-tick interval is set and if valid
    if t_int:
        try:
            daterange = pd.date_range(stime, etime+pd.Timedelta('1d'), freq=t_int, normalize=True)
            daterange = [t for t in daterange if t>=stime if t<=etime]
            key = translate_daterange_intervals(daterange, t_keys)
        except ValueError:
            print('Set "t_int" keyword smaller than time series length')
    else: # automatically choose x-tick interval
        key, daterange = x_tick_locs(stime, etime)

    # ===== x-tick location =====
    x_tickloc = [UTCDateTime(t).timestamp for t in daterange]

    # ===== x-tick format =====
    key_idx = t_keys.index(key) # get index of key in list
    # check if x-tick intervals are over a superior unit (eg. minute x-ticks, but crossing a full hour)
    same_superior = t_array(t_bins[0])[:key_idx] == t_array(t_bins[-1])[:key_idx]
    x_labels_fmt(key, same_superior)
    x_tickformat = x_labels_fmt(key, same_superior)[0]
    x_ticks_str = [t.strftime('{}'.format(x_tickformat)) for t in daterange]

    # set x-axis label
    x_label_time = stime.strftime('{}'.format(x_labels_fmt(key, same_superior)[1][1]))
    x_label = 'Time (UTC)  {} {}  {}'.format(x_labels_fmt(key, same_superior)[1][0],
                                            x_label_time,
                                            x_labels_fmt(key, same_superior)[1][2])
    return x_tickloc, x_ticks_str, x_label

# plot the spectrogram
def plot_spectro(Pxx, freqs, bins, stats, 
                 t_int=False,
                 cmap="viridis",
                 vmax=None,
                 vmin=None, # matplotlib color map
                 ylim=None, 
                 yscale='linear',
                 **kwargs):
    '''plot spectrogram from pre-calculated values in calc_spec'''    
    fig = plt.figure(figsize=(6.4*2,4.8))
    ax = fig.add_axes([0.125, 0.125, 0.76, 0.6])      # main spectrogram
    ax_cbar = fig.add_axes([0.895, 0.125, 0.02, 0.6]) # colorbar
    
    # apply the ylim here - directly reduce the data
    if ylim:
        idx_low = (np.abs(freqs - ylim[0])).argmin()
        idx_high = (np.abs(freqs - ylim[1])).argmin()
        Pxx_cut = Pxx[idx_low:idx_high,:]
        freqs_cut = freqs[idx_low:idx_high]
    else:
        Pxx_cut = Pxx
        freqs_cut = freqs
    
    im = ax.imshow(10*np.log10(Pxx_cut), 
                   aspect='auto', 
                   origin='lower', 
                   cmap=cmap,
                   extent=[stats.starttime.timestamp, stats.endtime.timestamp, freqs_cut[0], freqs_cut[-1]],
                   vmax=vmax, 
                   vmin=vmin)

    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.ax.locator_params(nbins=5)
    cbar.set_label('Power Spectral Density [dB]', fontsize=12) #colorbar label
    
    # set the x-ticks
    x_tickloc, x_ticks_str, x_label = nice_x_axis(stats, bins, t_int=t_int) # functions to plot nice x-axis
    ax.set_xticks(x_tickloc)
    ax.set_xticklabels(x_ticks_str)
    ax.set_xlabel(x_label, fontsize=12) 
    
    ax.set_ylabel('Frequency [Hz]', fontsize=12)
    ax.set_yscale(yscale)
    ax.set_ylim(ylim)
    return fig



def stretch(wave1, wave2, time, maxshift=0, min_ratio=0.1, max_ratio=2):

    interp_f = interp1d(time, wave2, bounds_error=False, fill_value=0.)
    n1 = np.sum(np.square(wave1))
    dt = time[1] - time[0]
    cc = 0
    relative_ratio = 1
    npts = len(time)

    for ratio in np.arange(min_ratio, max_ratio, 0.01):
        dt_new = dt / ratio
        time_new = np.arange(time[0], time[-1], dt_new)
        wave_new = interp_f(time_new)
        
        n2 = np.sum(np.square(wave_new))
        corr = sgn.correlate(wave1, wave_new) / np.sqrt(n1 * n2)

        l_maxshift = min(len(wave_new), maxshift)
        r_maxshift = min(len(wave1), maxshift)

        st_pt = len(wave_new) - l_maxshift
        en_pt = len(wave_new) + r_maxshift+1

        cc_best = np.nanmax(corr[st_pt: en_pt])

        if cc < cc_best:
            cc = cc_best
            relative_ratio = ratio

    dt_new = dt / relative_ratio
    time_new = np.arange(time[0], time[-1], dt_new)
    wave_new = interp_f(time_new)
    
    
    return wave_new, np.arange(len(wave_new))*dt, relative_ratio, cc


def stretch_distribution(wave1, wave2, time, maxshift=0, min_ratio=0.1, max_ratio=2):

    stretch_range = np.arange(min_ratio, max_ratio, 0.01)
    
    interp_f = interp1d(time, wave2, bounds_error=False, fill_value=0.)
    n1 = np.sum(np.square(wave1))
    dt = time[1] - time[0]
    cc = np.zeros(len(stretch_range), dtype = np.float32)
    npts = len(time)

    for i, ratio in enumerate(stretch_range):
        dt_new = dt / ratio
        time_new = np.arange(time[0], time[-1], dt_new)
        wave_new = interp_f(time_new)
        
        n2 = np.sum(np.square(wave_new))
        corr = sgn.correlate(wave1, wave_new) / np.sqrt(n1 * n2)

        l_maxshift = min(len(wave_new), maxshift)
        r_maxshift = min(len(wave1), maxshift)

        st_pt = len(wave_new) - l_maxshift
        en_pt = len(wave_new) + r_maxshift+1

        cc[i] = np.nanmax(corr[st_pt: en_pt])
    
    return stretch_range, cc


def fk_filter_2cones(vsp, w1=0, w2=0, cone1=False, cone2=False):
    n1, n2 = vsp.shape
    nf = next_power_of_2(n1)
    nk = next_power_of_2(n2)

    nf2=int(nf/2)
    nk2=int(nk/2)
    
    fk2d = np.fft.fft2(vsp, s=(nf,nk))
    fk2d = np.fft.fftshift(fk2d, axes=(-2,-1))
    
    nw1 = int(np.ceil(w1*nk))
    nw2 = int(np.ceil(w2*nf))

    mask1=np.ones((nf,nk), dtype=np.float64)
    mask2=np.ones((nf,nk), dtype=np.float64)

    if cone1:
        for j in np.arange(nk2-nw1, nk2+1):
            th1 = int((j-nk2+nw1) * nf2/nw1)

            mask1[:th1, j] = 0
            mask1[nf-th1:, j] = 0
            mask1[:th1, nk-j] = 0
            mask1[nf-th1:, nk-j] = 0

    if cone2:
        for j in np.arange(0, nk2):
            th2 = int(nf2 - (nw2/nk2)*(nk2-j))
            mask2[th2:nf-th2+1, j] = 0
            if j != 0:
                mask2[th2:nf-th2+1, nk-j] = 0


    mask = mask2*mask1
    
    filtered_2d = fk2d * mask
    tmp = np.fft.ifftshift(filtered_2d)
    output = np.fft.ifft2(tmp, s=(nk,nf), axes=(-1, -2))
    
    return output[:n1,:n2], filtered_2d, fk2d


def peak_dvv(new_peaks, ax, thrs=0.3):
    f = interp1d(np.nonzero(new_peaks>0)[0], new_peaks[new_peaks>0], bounds_error=False, fill_value="extrapolate", kind='linear')
    new_peaks_iloc = f(np.arange(len(new_peaks)))
    grad_iloc = np.abs(np.diff(new_peaks_iloc))
    grad_peaks_iloc, _ = find_peaks(grad_iloc, prominence=0.05)
    grad_peaks_iloc += 1
    preserve_inds = [np.arange(50, 60), np.arange(201, 299)]
    for (ini, iend) in [[0, 50], [60, 201], [299, len(new_peaks_iloc)]]:
        grad_peaks_interval = grad_peaks_iloc[(grad_peaks_iloc > ini) & (grad_peaks_iloc < iend)]-ini
        if len(grad_peaks_interval) <= 1:
            continue
        data_segs = np.split(new_peaks_iloc[ini:iend], grad_peaks_interval)
        # print('Len of segments'+str([len(seg) for seg in data_segs]))
        ind_segs = np.split(np.arange(ini, iend), grad_peaks_interval)
        for i in range(len(data_segs)):
            if len(data_segs[i]) < 3:
                continue
            if np.abs(data_segs[i][0]-np.mean(data_segs[i][1:])) > 0.1:
                data_segs[i] = data_segs[i][1:]
                ind_segs[i] = ind_segs[i][1:]
            if np.abs(data_segs[i][-1]-np.mean(data_segs[i][:-1])) > 0.1:
                data_segs[i] = data_segs[i][:-1]
                ind_segs[i] = ind_segs[i][:-1]
        ref_seg_id = np.argmax([len(seg) for seg in data_segs])
        mean_segs = np.array([np.mean(seg) for seg in data_segs])
        preserve_seg_id = [ref_seg_id]
        num_iter = 0
        while True:
    
            coefficients = np.polyfit(np.hstack([ind_segs[i] for i in preserve_seg_id]), 
                np.hstack([data_segs[i] for i in preserve_seg_id]), 1)
            polynomial = np.poly1d(coefficients)
            mean_distance = np.array([np.sqrt(np.mean(np.square(polynomial(ind_segs[i])-data_segs[i]))) for i in range(len(data_segs))])

            # combine nearby segments
            if num_iter == 0:
                combine_ids = np.argsort(mean_distance)[:2]
                # print(combine_ids)
            else:
                combine_ids = np.nonzero(mean_distance < thrs * (np.amax(mean_segs)-np.amin(mean_segs)))[0]
    
            
            if len(combine_ids) == len(preserve_seg_id):
                break
            preserve_seg_id = combine_ids
            num_iter += 1
        preserve_inds.extend([ind_segs[i] for i in preserve_seg_id])
        # ax.plot(np.arange(ini, iend), polynomial(np.arange(ini, iend)), 'g')
    # interpolate
    preserve_inds = np.hstack(preserve_inds)
    f = interp1d(preserve_inds, new_peaks_iloc[preserve_inds], bounds_error=False, fill_value="extrapolate", kind='linear')
    new_peaks_iloc[:] = f(np.arange(len(new_peaks_iloc)))

    ### remove scatter points 
    # while True:
    pks, _ = find_peaks(np.abs(np.diff(new_peaks_iloc)), prominence=1)
    # if len(pks) == 0:
    #     break
    
    seg_inds = np.split(np.arange(len(new_peaks_iloc)), pks+1)
    valid_id = []
    for _ in seg_inds:
        if len(_) > 5:
            valid_id.extend(_)
    # if len(valid_id) == len(new_peaks_iloc):
    #     break
    f = interp1d(valid_id, new_peaks_iloc[valid_id], bounds_error=False, fill_value="extrapolate", kind='linear')
    new_peaks_iloc[:] = f(np.arange(len(new_peaks_iloc)))

    
    ax.plot(x[1:], np.abs(np.diff(new_peaks_iloc))*10, 'purple')
    
    ax.plot(x, new_peaks_iloc, 'g')

    return new_peaks_iloc


def multi_bounds(iloc, peaks, input_image, ax, cc_dvv, new_peaks):
    
    dvv_ind = np.zeros(len(x), dtype=np.int32)

    #####################################################################1
    jump_point = 1
    for percent in [80,75,70,65,60,55,50,45,40,35,30,25,20,15,10,5]:
        jump = np.percentile(peaks[iloc,0:100], percent) - np.percentile(peaks[iloc,0:100], percent-5)
        if jump > 0.2:
            jump_point = percent
    
    upper = int(np.percentile(peaks[iloc,0:100], 99) / 0.01 - 75)
    lower = int(np.percentile(peaks[iloc,0:100], jump_point) / 0.01 - 85)
    upper = max(upper, 40)
    lower = max(lower, 0)
    
    dvv_ind[0:100] = np.argmax(input_image[lower:upper, 0:100], axis=0) + lower
    
    for i in np.arange(0,100):
        new_peaks[iloc,i] = stretch_range[dvv_ind[i]]
        cc_dvv[iloc,i] = input_image[dvv_ind[i], i]  
        
    ax.hlines(y=lower, color='w', linestyle='dotted', xmin=0, xmax=0.2)
    ax.hlines(y=upper, color='k', linestyle='dotted', xmin=0, xmax=0.2)
     #####################################################################2
    jump_point = 1
    for percent in [80,75,70,65,60,55,50,45,40,35,30,25,20,15,10,5]:
        jump = np.percentile(peaks[iloc,100:170], percent) - np.percentile(peaks[iloc,100:170], percent-5)
        if jump > 0.1:
            jump_point = percent
    
    ## set bounds using 1st round dvv
    upper = int(np.percentile(peaks[iloc,100:170], 99) / 0.01 - 75)
    lower = int(np.percentile(peaks[iloc,100:170],  jump_point) / 0.01 - 85)
    upper = max(upper, 40)
    lower = max(lower, 0)    

    dvv_ind[100:170] = np.argmax(input_image[lower:upper, 100:170], axis=0) + lower
    
    for i in np.arange(100,170):
        new_peaks[iloc,i] = stretch_range[dvv_ind[i]]
        cc_dvv[iloc,i] = input_image[dvv_ind[i], i]   
            
    ax.axhline(y=lower, color='w', linestyle='dotted', xmin=0.2, xmax=0.35)
    ax.axhline(y=upper, color='k', linestyle='dotted', xmin=0.2, xmax=0.35)
    #####################################################################3
    jump_point = 1
    for percent in [80,75,70,65,60,55,50,45,40,35,30,25,20,15,10,5]:
        jump = np.percentile(peaks[iloc,170:250], percent) - np.percentile(peaks[iloc,170:250], percent-5)
        if jump > 0.2:
            jump_point = percent
    
    ## set bounds using 1st round dvv
    upper = int(np.percentile(peaks[iloc,170:250], 99) / 0.01 - 75)
    lower = int(np.percentile(peaks[iloc,170:250],  jump_point) / 0.01 - 85)
    upper = max(upper, 40)
    lower = max(lower, 0)
    dvv_ind[170:250] = np.argmax(input_image[lower:upper, 170:250], axis=0) + lower
    
    for i in np.arange(170,250):
        new_peaks[iloc,i] = stretch_range[dvv_ind[i]]
        cc_dvv[iloc,i] = input_image[dvv_ind[i], i]   
            
    ax.axhline(y=lower, color='w', linestyle='dotted', xmin=0.35, xmax=0.52)
    ax.axhline(y=upper, color='k', linestyle='dotted', xmin=0.35, xmax=0.52)
    #####################################################################4
    jump_point = 1
    for percent in [80,75,70,65,60,55,50,45,40,35,30,25,20,15,10,5]:
        jump = np.percentile(peaks[iloc,250:368], percent) - np.percentile(peaks[iloc,250:368], percent-5)
        if jump > 0.15:
            jump_point = percent
    
    ## set bounds using 1st round dvv
    upper = int(np.percentile(peaks[iloc,250:368], 99) / 0.01 - 75)
    lower = int(np.percentile(peaks[iloc,250:368],  jump_point) / 0.01 - 85)
    upper = max(upper, 40)
    lower = max(lower, 0)
    dvv_ind[250:368] = np.argmax(input_image[lower:upper, 250:368], axis=0) + lower
    
    for i in np.arange(250,368):
        new_peaks[iloc,i] = stretch_range[dvv_ind[i]]
        cc_dvv[iloc,i] = input_image[dvv_ind[i], i]   
            
    ax.axhline(y=lower, color='w', linestyle='dotted', xmin=0.52, xmax=0.76)
    ax.axhline(y=upper, color='k', linestyle='dotted', xmin=0.52, xmax=0.76)
     #####################################################################5
    jump_point = 1
    for percent in [80,75,70,65,60,55,50,45,40,35,30,25,20,15,10,5]:
        jump = np.percentile(peaks[iloc,368:], percent) - np.percentile(peaks[iloc,368:], percent-5)
        if jump > 0.2:
            jump_point = percent
    
    ## set bounds using 1st round dvv
    upper = int(np.percentile(peaks[iloc,368:], 99) / 0.01 - 75)
    lower = int(np.percentile(peaks[iloc,368:],  jump_point) / 0.01 - 85)
    upper = max(upper, 40)
    lower = max(lower, 0)
    dvv_ind[368:] = np.argmax(input_image[lower:upper, 368:], axis=0) + lower
    
    for i in np.arange(368,len(x)):
        new_peaks[iloc,i] = stretch_range[dvv_ind[i]]
        cc_dvv[iloc,i] = input_image[dvv_ind[i], i]   
            
    ax.axhline(y=lower, color='w', linestyle='dotted', xmin=0.76)
    ax.axhline(y=upper, color='k', linestyle='dotted', xmin=0.76)
    #####################################################################
    ax.scatter(x, new_peaks[iloc] / 0.01 - 75, cmap='viridis', c=cc_dvv[iloc], s=10, marker='o', vmin=0.3, vmax=0.5)

    return new_peaks[iloc]  / 0.01 -75  



def compute_misfit(a, b, tillage_interpolated, tire_interpolated, dvv_variability):
    scaled_mechanical = np.power(tillage_interpolated, a) * np.power(tire_interpolated, b)
    scaled_mechanical = scaled_mechanical/np.std(scaled_mechanical)
    scaled_variability =dvv_variability/np.std(dvv_variability)
    correlation = np.corrcoef(scaled_variability, scaled_mechanical)[0, 1]
    mse = mean_squared_error(scaled_variability, scaled_mechanical)

    return correlation, mse, scaled_mechanical, scaled_variability