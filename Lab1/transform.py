#!/usr/bin/env python3

import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def load_plz(pattern):
    
    files = sorted(glob.glob(pattern))
    files = map(fits.open, files)
    files = [file[0].data.flatten() for file in files]
    return np.array(files)
    
def lazy_load_plz(pattern):
    
    files = sorted(glob.glob(pattern))
    files = map(fits.open, files)

def num_sd(arr):
    
    return np.abs(arr - arr.mean()) / arr.std()

def sigma_clip(img, mask):
    
    img *= mask
    img += img.median() * ~mask

darks = load_plz("data/*DARK*")
dark_master = np.median(darks, axis=0)

flats = load_plz("data/*FLAT*")
flat_master = np.median(flats, axis=0)
flat_master /= np.median(flat_master)

dark_sigma = num_sd(dark_master)
dark_mask = dark_sigma < 3

flat_sigma = num_sd(flat_master)
flat_mask = flat_sigma < 3

big_mask = flat_mask * dark_mask

def transform(img):
    
    img -= dark_master
    img /= flat_master
    sigma_clip(img, big_mask)

science_images = lazy_load_plz("data/Science_image*")

for i, hdul in enumerate(science_images):
    
    img = hdul[0].data
    transform(img)
    hdul.writeto(f"transformed_{i+1}.fits")
