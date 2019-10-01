#!/usr/bin/env python

import glob
import numpy as np
import pandas as pd
from transform import lazy_load_plz

def load_cat(filename):
    return pd.read_csv(filename, sep='\s+', skiprows=5)

def load_time(file_num):
    file_name = "Science_image." + file_num + ".FIT"
    file, = lazy_load_plz([file_name])
    time_string = file[0].header["TIME-OBS"]
    h, m, s = map(float, time_string.strip().split(':'))
    seconds = 3600*h + 60*m + s
    return seconds


kitties = glob.glob("data/*.cat")
i = 0
for cat_name in kitties:
    kitty = load_cat(cat_name)
    cat_num = cat_name.split('.')[-3]
    time = load_time(cat_num)
    i = i+1
    kitty = kitty.sort_values(by=3, ascending=False) #column 3 is flux_aper
    if i == 1:
        print(kitty)


    
    
    
