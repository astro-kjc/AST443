#!/usr/bin/env python3

import glob
import os
import numpy as np
import pandas as pd
from transform import lazy_load_plz
import matplotlib.pyplot as plt

def load_cat(filename):
    return pd.read_csv(filename, sep='\s+', skiprows=5, header=None, names=['Num','Ra', 'Dec',
            'Flux', 'Flux_err'])

def load_time(file_num):
    file_name = "Science_image." + file_num + ".FIT"
    file, = lazy_load_plz("data/" +file_name)
    time_string = file[0].header["TIME-OBS"]
    h, m, s = map(float, time_string.strip().split(':'))
    seconds = 3600*h + 60*m + s
    return seconds

def is_kinda_in(a, b, eps):

    def map_one(el):

        return (np.abs(b - el) < eps).any()

    return a.apply(map_one)

kitties = glob.glob("data/*.cat")
kitties = sorted(kitties)

t = []
time_series = []
NUMREF = 15
reference_stars = None
cat_less = []

for i, cat_name in enumerate(kitties):

    kitty = load_cat(cat_name)
    cat_num = cat_name.split('.')[-3]
    time = load_time(cat_num)

    if reference_stars is None:

        kitty.sort_values(by='Flux', ascending=False, inplace=True)

        mask = kitty['Ra'] >= 300
        mask &= kitty['Ra'] <= 300.25
        mask &= kitty['Dec'] >= 22.5
        mask &= kitty['Dec'] <= 22.8

        masked_kitty = kitty[mask][:NUMREF]
        masked_kitty.sort_values(by='Ra', inplace=True)
        reference_stars = np.array([masked_kitty['Ra'], masked_kitty['Dec']])
    if int(cat_num) > 1403 and int(cat_num) < 1562:
        epsilon = 8e-4
    else:
        epsilon = 5e-4
    kitty.sort_values(by='Ra', inplace=True)
    mask = is_kinda_in(kitty['Ra'], reference_stars[0], epsilon)
    mask &= is_kinda_in(kitty['Dec'], reference_stars[1], epsilon)
    cut_kitty = kitty[mask]

    if len(cut_kitty) == NUMREF:
        time_series.append(cut_kitty)
        t.append(time)
    elif len(cut_kitty) < 10:
        cat_less.append(int(cat_num))
print(len(cat_less))
print(cat_less)

flux = np.array([list(df['Flux']) for df in time_series])
flux_err = np.array([list(df['Flux_err']) for df in time_series])

mu = np.mean(flux, axis=0)
flux /= mu
flux_err /= mu

flux = flux.T
flux_err = flux_err.T

if not os.path.isdir("lightcurves"):
    os.mkdir("lightcurves")

np.savetxt("lightcurves/ref_stars.csv", reference_stars, delimiter=",")
np.savetxt("lightcurves/flux.csv", flux, delimiter=",")
np.savetxt("lightcurves/flux_err.csv", flux_err, delimiter=",")
np.savetxt("lightcurves/times.csv", t, delimiter=",")

# I tried a bunch of subplots, but they weren't really readable

for i in range(len(flux)):

    ra, dec = reference_stars[0][i], reference_stars[1][i]
    plt.errorbar(t, flux[i], flux_err[i], fmt='.',
            label=rf"$\alpha$ = {ra:.2f}$^\circ$, $\delta$ = {dec:.2f}$^\circ$")
    plt.ylim(min(flux[i].min(), 0.8), max(flux[i].max(), 1.2))
    plt.legend()
    plt.savefig(f"lightcurves/lightcurve_{i}.png")
    plt.gcf().clear()
