#!/usr/bin/env python

import glob
import numpy as np
import pandas as pd
from transform import lazy_load_plz
import matplotlib.pyplot as plt

def load_cat(filename):
    return pd.read_csv(filename, sep='\s+', skiprows=5, header=None, names=['Num','Ra', 'Dec', 'Flux', 'Flux_err'])

def load_time(file_num):
    file_name = "Science_image." + file_num + ".FIT"
    file, = lazy_load_plz("data/" +file_name)
    time_string = file[0].header["TIME-OBS"]
    h, m, s = map(float, time_string.strip().split(':'))
    seconds = 3600*h + 60*m + s
    return seconds

t=[]
time_series=[]
reference_stars = None
kitties = glob.glob("data/*.cat")
for cat_name in kitties:
    kitty = load_cat(cat_name)
    cat_num = cat_name.split('.')[-3]
    time = load_time(cat_num)

    if reference_stars is None:
        kitty = kitty.sort_values(by='Flux', ascending=False)
        reference_stars = kitty['Num'][:15]
    
    cut_kitty = kitty[kitty['Num'].isin(reference_stars)].sort_values(by="Num")

    time_series.append(cut_kitty)
    t.append(time)

print('-'*50)
print(np.asarray(time_series[0]['Flux']).T)
print('-'*50)

flux = [list(df['Flux']) for df in time_series]
flux_err = [list(df['Flux_err']) for df in time_series]

flux = np.array(flux, dtype=np.float64).T
flux_err = np.array(flux_err, dtype=np.float64).T

print(flux)

for i in range(len(flux)):
    print(flux[i])
    print(t)
    plt.plot(t,flux[i], yerr=flux_err[i])

plt.show()


