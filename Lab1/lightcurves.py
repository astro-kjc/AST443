#!/usr/bin/env python3

import glob
import os
import numpy as np
import pandas as pd
from transform import lazy_load_plz
import matplotlib.pyplot as plt

def load_cat(filename):
    return pd.read_csv(filename, sep=r'\s+', skiprows=5, header=None, names=['Num','Ra', 'Dec',
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
    
def distances(starcoords, catframe):
    
    alpha, delta = starcoords
    dra = (alpha - catframe['Ra']) * np.cos(delta)
    ddec = delta - catframe['Dec']
    return np.sqrt(dra**2 + ddec**2)
    
def fuzzy_match(starcoords, catframe, eps=1e-3):
    
    gamma = distances(starcoords, catframe)
    indx = gamma.idxmin()
    
    if gamma[indx] <= eps:
        return catframe.loc[indx]
    return None
    
def get_field(row_list, field):
    
    return [row[field] for row in row_list]

if __name__ == "__main__":

    kitties = glob.glob("data/*.cat")
    kitties = sorted(kitties)

    t = np.zeros(len(kitties), dtype=np.float64)
    NUMREF = 15
    reference_stars = None
    time_series = [list() for i in range(NUMREF)]
    indices = [list() for i in range(NUMREF)]

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
            reference_stars = np.array([masked_kitty['Ra'], masked_kitty['Dec']]).T
            
        if len(kitty) <= 0: continue
            
        for j, starcoords in enumerate(reference_stars):
            
            match = fuzzy_match(starcoords, kitty)
            if match is not None:
                time_series[j].append(match)
                indices[j].append(i)
                
        t[i] = time
        
        prog = i / len(kitties)
        numspace = int(50*(1-prog))
        pbar = "[" + "#" * (50-numspace) + " " * numspace + "]"
        end = "" if i < len(kitties) - 1 else "\n"
        print(f"\r{pbar} {100*prog:.0f}%", end=end)

    flux = (get_field(row_list, "Flux") for row_list in time_series)
    flux = list(map(np.array, flux))

    flux_err = (get_field(row_list, "Flux_err") for row_list in time_series)
    flux_err = list(map(np.array, flux_err))

    mu = [flux_arr.mean() for flux_arr in flux]

    flux = [flux[i] / mu[i] for i in range(len(flux))]
    flux_err = [flux_err[i] / mu[i] for i in range(len(flux_err))]

    if not os.path.isdir("lightcurves"):
        os.mkdir("lightcurves")

    for i in range(NUMREF):
        if not os.path.isdir(f"lightcurves/star_{i}"):
            os.mkdir(f"lightcurves/star_{i}")

    np.savetxt("lightcurves/ref_stars.csv", reference_stars, header="RA Dec", delimiter=",")
    np.savetxt("lightcurves/times.csv", t, delimiter=",")

    for i in range(NUMREF):
        np.savetxt(f"lightcurves/star_{i}/flux_{i}.csv", flux[i], delimiter=",")
        np.savetxt(f"lightcurves/star_{i}/flux_err_{i}.csv", flux_err[i], delimiter=",")
        np.savetxt(f"lightcurves/star_{i}/indices_{i}.csv", indices[i], delimiter=",")

    # I tried a bunch of subplots, but they weren't really readable

    for i in range(NUMREF):

        ra, dec = reference_stars[i]
        plt.errorbar(t[indices[i]], flux[i], flux_err[i], fmt='.',
                label=rf"$\alpha$ = {ra:.2f}$^\circ$, $\delta$ = {dec:.2f}$^\circ$")
        plt.xlabel("Time Since Midnight UTC [s]")
        plt.ylabel(r"$\frac{f}{\langle f \rangle}$")
        plt.legend()
        plt.savefig(f"lightcurves/star_{i}/lightcurve_{i}.png")
        plt.gcf().clear()
