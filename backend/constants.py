# author: carolina ferrari

import numpy as np

g = 9.80665  # gravitational acceleration in m/s² (standard value)
rho = 1.225  # air density in kg/m³ (at sea level, 15°C)
cl = 3.12    # coefficient of lift (dimensionless)
cd = 1.31    # coefficient of drag (dimensionless)
fa = 1.022   # frontal area in m²
wb = 1.550   # wheelbase in meters (1550 mm → 1.550 m)
ft = 1.150   # front track in meters (1150 mm → 1.150 m)
rt = 1.150   # rear track in meters (1150 mm → 1.150 m)
rl = 0.226   # rolling radius in meters (17.8 * 25.4 / 2 mm → 0.226 m)
wf = (97.3 + 68) * g  # front axle weight in Newtons (kg → N)
wr = (118.4 + 68) * g  # rear axle weight in Newtons (kg → N)
w = wf + wr  # total vehicle weight in Newtons
steering_ratio = 0.22  # steering ratio (dimensionless)
gear_reduction = 4.15 * 2.111 * np.array([1.938, 1.556, 1.348, 1.208, 1.095])  # gear reduction ratios (dimensionless)