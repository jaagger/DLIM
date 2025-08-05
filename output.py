import os
import warnings
from matplotlib import cm
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt

from funtion_Ne import *

lat=20
lon=20
lt=12
doy = 103
kp = 0.7
f107 = 75.3
symh = -16

NEnow,alt=ne_profile_function(lat, lon, lt, doy, f107, kp, symh)

print(NEnow,alt)