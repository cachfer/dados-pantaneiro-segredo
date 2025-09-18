# author: carolina ferrari

import numpy as np

# IMPORTED VARIABLES UNITS (as received from FTManager)
IMPORTED_UNITS = {
    'time': 's',                    # Time in seconds
    'rpm': 'rpm',                   # Engine RPM
    'tps': '%',                     # Throttle position sensor percentage
    'engine_temperature': '°C',     # Engine temperature in Celsius
    'fuel_flow': 'lb/h',            # Fuel flow rate in pounds per hour
    'oil_pressure': 'bar',          # Oil pressure in bar
    'oil_temperature': '°C',        # Oil temperature in Celsius
    'battery_voltage': 'V',         # Battery voltage in volts
    'brake_pressure_rear': 'psi',   # Rear brake pressure in psi
    'brake_pressure_front': 'psi',  # Front brake pressure in psi
    'gear': 'count',                # Gear number (count)
    'traction_speed': 'km/h',       # Vehicle speed in km/h
    'G_longitudinal': 'g',          # Longitudinal G force
    'G_lateral': 'g',               # Lateral G force
    'general_probe': 'lambda',      # General lambda probe reading
    'mesh_target': 'lambda',        # Lambda target value
    'wps': '°',                     # Wheel position sensor in degrees
    'FL_Damper': 'mm',              # Front left damper position in mm
    'FR_Damper': 'mm',              # Front right damper position in mm
    'RL_Damper': 'mm',              # Rear left damper position in mm
    'RR_Damper': 'mm',              # Rear right damper position in mm
    '2step': 'on/off',              # 2-step activation (on/off)
}

# CONVERSION FACTORS TO SI UNITS
CONVERSION_FACTORS = {
    'lbh_to_kgs': 0.453592 / 3600,  # lb/h to kg/s
    'psi_to_pa': 6894.76,           # psi to Pa
    'bar_to_pa': 100000,            # bar to Pa
    'kmh_to_ms': 1/3.6,             # km/h to m/s
    'deg_to_rad': np.pi/180,        # degrees to radians
    'mm_to_m': 0.001,               # mm to meters
    'celsius_to_kelvin': lambda x: x + 273.15,  # °C to K
    'percent_to_ratio': 0.01,       # % to ratio (0-1)
    'rpm_to_hz': 1/60,              # RPM to Hz
    'hp_to_w': 745.7,               # HP to W
    'g_to_ms2': 9.80665,            # g-force to m/s²
}

# CONVERSION FACTORS FROM SI UNITS BACK TO ORIGINAL
CONVERSION_FROM_SI = {
    'kgs_to_lbh': 3600 / 0.453592,  # kg/s to lb/h
    'pa_to_psi': 1/6894.76,         # Pa to psi
    'pa_to_bar': 1/100000,          # Pa to bar
    'ms_to_kmh': 3.6,               # m/s to km/h
    'rad_to_deg': 180/np.pi,        # radians to degrees
    'm_to_mm': 1000,                # meters to mm
    'kelvin_to_celsius': lambda x: x - 273.15,  # K to °C
    'ratio_to_percent': 100,        # ratio to %
    'hz_to_rpm': 60,                # Hz to RPM
    'w_to_hp': 1/745.7,            # W to HP
    'ms2_to_g': 1/9.80665,         # m/s² to g-force
}

def convert_to_si(data_dict, variable_name, value):
    if variable_name in ['brake_pressure_front', 'brake_pressure_rear', 'brake_pressure_total']:
        return value * CONVERSION_FACTORS['psi_to_pa']
    elif variable_name in ['oil_pressure', 'fuel_pressure', 'low_oil_pressure_limit']:
        return value * CONVERSION_FACTORS['bar_to_pa']
    elif variable_name == 'traction_speed':
        return value * CONVERSION_FACTORS['kmh_to_ms']
    elif variable_name in ['engine_temperature', 'oil_temperature']:
        return CONVERSION_FACTORS['celsius_to_kelvin'](value)
    elif variable_name in ['wps', 'steering_angle', 'delta_acker']:
        return value * CONVERSION_FACTORS['deg_to_rad']
    elif variable_name in ['FL_Damper', 'FR_Damper', 'RL_Damper', 'RR_Damper']:
        return value * CONVERSION_FACTORS['mm_to_m']
    elif variable_name == 'fuel_flow':
        return value * CONVERSION_FACTORS['lbh_to_kgs']
    elif variable_name == 'rpm':
        return value * CONVERSION_FACTORS['rpm_to_hz']
    elif variable_name == 'tps':
        return value * CONVERSION_FACTORS['percent_to_ratio']
    elif variable_name in ['G_lateral', 'G_longitudinal', 'G_combined']:
        return value * CONVERSION_FACTORS['g_to_ms2']
    else:
        return value  # Already in SI units or dimensionless

def convert_from_si(data_dict, variable_name, value):
    if variable_name in ['brake_pressure_front', 'brake_pressure_rear', 'brake_pressure_total']:
        return value * CONVERSION_FROM_SI['pa_to_psi']
    elif variable_name in ['oil_pressure', 'fuel_pressure', 'low_oil_pressure_limit']:
        return value * CONVERSION_FROM_SI['pa_to_bar']
    elif variable_name == 'traction_speed':
        return value * CONVERSION_FROM_SI['ms_to_kmh']
    elif variable_name in ['engine_temperature', 'oil_temperature']:
        return CONVERSION_FROM_SI['kelvin_to_celsius'](value)
    elif variable_name in ['wps', 'steering_angle', 'delta_acker']:
        return value * CONVERSION_FROM_SI['rad_to_deg']
    elif variable_name in ['FL_Damper', 'FR_Damper', 'RL_Damper', 'RR_Damper']:
        return value * CONVERSION_FROM_SI['m_to_mm']
    elif variable_name == 'fuel_flow':
        return value * CONVERSION_FROM_SI['kgs_to_lbh']
    elif variable_name == 'rpm':
        return value * CONVERSION_FROM_SI['hz_to_rpm']
    elif variable_name == 'tps':
        return value * CONVERSION_FROM_SI['ratio_to_percent']
    elif variable_name in ['G_lateral', 'G_longitudinal', 'G_combined']:
        return value * CONVERSION_FROM_SI['ms2_to_g']
    elif variable_name == 'power':
        return value * CONVERSION_FROM_SI['w_to_hp']
    else:
        return value  # Already in original units or dimensionless

COLUMN_MAPPING = {
    'TIME': 'time',
    'RPM': 'rpm',
    'TPS': 'tps',
    'Temp._do_motor': 'engine_temperature',
    'Vazão_da_bancada_A': 'fuel_flow',
    'Pressão_de_Óleo': 'oil_pressure',
    'Temperatura_do_óleo': 'oil_temperature',
    'Tensão_da_Bateria': 'battery_voltage',
    'Pressão_da_embreagem': 'brake_pressure_rear',
    'Pressão_do_freio': 'brake_pressure_front',
    'Marcha': 'gear',
    'Velocidade_de_tração': 'traction_speed',
    'Força_G_aceleração': 'G_longitudinal',
    'Força_G_lateral': 'G_lateral',
    'Sonda_Geral': 'general_probe',
    'Alvo_do_malha_fechada': 'mesh_target',
    'WPS': 'wps',
    'Amortecedor_dianteiro_esquerdo': 'FL_Damper',
    'Amortecedor_dianteiro_direito': 'FR_Damper',
    'Amortecedor_traseiro_esquerdo': 'RL_Damper',
    'Amortecedor_traseiro_direito': 'RR_Damper',
    '2-step': '2step'
}

REQUIRED_COLUMNS = [
    'battery_voltage',
    'brake_pressure_front',
    'brake_pressure_rear',
    'engine_temperature',
    'fuel_flow',
    'fuel_pressure',
    'G_lateral',
    'G_longitudinal',
    'gear',
    'general_probe',
    'mesh_target',
    'time',
    'oil_temperature',
    'oil_pressure',
    'rpm',
    'tps',
    'traction_speed',
    'wps',
    'FL_Damper',
    'FR_Damper',
    'RL_Damper',
    'RR_Damper'
]

# thresholds and limits for calculations (ORIGINAL UNITS)
THRESHOLDS_ORIGINAL = {
    'brake_pressure_min': 15, # Minimum brake pressure for calculations (psi)
    'high_brake_pressure': 800, # High brake pressure threshold (psi)
    'high_lateral_g': 0.5, # Threshold for high lateral G force (g)
    'min_speed_aero': 36, # Minimum speed for aero calculations (km/h)
    'high_g_combined': 0.8, # Threshold for high combined G force (g)
    'min_traction_g': 0.2, # Minimum longitudinal G for traction (g)
    'high_tps': 95, # High throttle position threshold (%)
    'low_tps': 5, # Low throttle position threshold (%)
    'high_wps': 10, # High wheel position sensor threshold (deg)
    'rpm_threshold': 9500, # RPM threshold for oil pressure calculation (rpm)
    'low_fuel_pressure': 2.98, # Low fuel pressure threshold (bar)
    'low_battery': 12, # Low battery voltage threshold (V)
    'high_engine_temp': 105, # High engine temperature threshold (°C)
    'high_oil_temp': 120, # High oil temperature threshold (°C)
}

# thresholds and limits for calculations (SI UNITS)
THRESHOLDS = {
    'brake_pressure_min': THRESHOLDS_ORIGINAL['brake_pressure_min'] * CONVERSION_FACTORS['psi_to_pa'], # Pa
    'high_brake_pressure': THRESHOLDS_ORIGINAL['high_brake_pressure'] * CONVERSION_FACTORS['psi_to_pa'], # Pa
    'high_lateral_g': THRESHOLDS_ORIGINAL['high_lateral_g'] * CONVERSION_FACTORS['g_to_ms2'], # m/s²
    'min_speed_aero': THRESHOLDS_ORIGINAL['min_speed_aero'] * CONVERSION_FACTORS['kmh_to_ms'], # m/s
    'high_g_combined': THRESHOLDS_ORIGINAL['high_g_combined'] * CONVERSION_FACTORS['g_to_ms2'], # m/s²
    'min_traction_g': THRESHOLDS_ORIGINAL['min_traction_g'] * CONVERSION_FACTORS['g_to_ms2'], # m/s²
    'high_tps': THRESHOLDS_ORIGINAL['high_tps'] * CONVERSION_FACTORS['percent_to_ratio'], # ratio
    'low_tps': THRESHOLDS_ORIGINAL['low_tps'] * CONVERSION_FACTORS['percent_to_ratio'], # ratio
    'high_wps': THRESHOLDS_ORIGINAL['high_wps'] * CONVERSION_FACTORS['deg_to_rad'], # rad
    'rpm_threshold': THRESHOLDS_ORIGINAL['rpm_threshold'] * CONVERSION_FACTORS['rpm_to_hz'], # Hz
    'low_fuel_pressure': THRESHOLDS_ORIGINAL['low_fuel_pressure'] * CONVERSION_FACTORS['bar_to_pa'], # Pa
    'low_battery': THRESHOLDS_ORIGINAL['low_battery'], # V (already SI)
    'high_engine_temp': CONVERSION_FACTORS['celsius_to_kelvin'](THRESHOLDS_ORIGINAL['high_engine_temp']), # K
    'high_oil_temp': CONVERSION_FACTORS['celsius_to_kelvin'](THRESHOLDS_ORIGINAL['high_oil_temp']), # K
}
