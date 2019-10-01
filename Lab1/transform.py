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
    return files

def num_sd(arr):
    
    return np.abs(arr - arr.mean()) / arr.std()
    
def apply_mask(arr, mask):
    
    arr = arr.astype(np.float32)
    
    arr *= mask
    arr[1:-1, 1:-1] += ~mask[1:-1, 1:-1] * 0.25 * \
                      (arr[:-2, 1:-1] + arr[2:, 1:-1] +
                       arr[1:-1, :-2] + arr[1:-1, 2:])
    return arr

if __name__ == "__main__":
    darks = load_plz("data/*DARK*")
    dark_master = np.median(darks, axis=0)

    flats = load_plz("data/*FLAT*")
    flat_master = np.median(flats, axis=0)
    flat_master /= np.median(flat_master)

    dark_sigma = num_sd(dark_master)
    dark_mask = dark_sigma < 5

    flat_sigma = num_sd(flat_master)
    flat_mask = flat_sigma < 5

    big_mask = flat_mask * dark_mask
    dim = int(np.sqrt(len(big_mask)))
    big_mask2d = np.reshape(big_mask, (dim, dim))

    def transform(img):
    
        img -= dark_master
        img /= flat_master

    science_images = lazy_load_plz("data/Science_image*")

    """
    for i, hdul in enumerate(science_images):
    
        img = hdul[0].data
        transform(img)
        hdul.writeto(f"data/transformed_{i+1:04}.fits")
        """
        
    for i, hdul in enumerate(science_images):
            
        hdul[0].data = apply_mask(hdul[0].data, big_mask2d)
        hdul.writeto(f"data/mapped.{i+1:04}.fits")
        
        mask_hdu = fits.PrimaryHDU(big_mask2d.astype(np.int32))
        mask_hdul = fits.HDUList([mask_hdu])
        mask_hdul.writeto(f"data/bad_pixels.{i+1:04}.fits")
            
